import torch
import torch.nn as nn
from einops import rearrange
from scipy.ndimage import gaussian_filter


# -----------------------------
# Helpers / Building Blocks
# -----------------------------
class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        logits = self.fc(x)          # [B, V]
        return self.softmax(logits)  # [B, V]


class Expert(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b (d h w) c")  # [B, N, C]
        x = self.norm(x)
        x = self.attn(x)
        x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
        return x


# -----------------------------
# Fusion Block
# -----------------------------
class Residual3DFusionBlock(nn.Module):
    def __init__(self, in_channels: int, view_num: int, apply_attention: bool = False):
        super().__init__()
        self.view_num = view_num
        self.gating_network = GatingNetwork(in_channels * view_num, view_num)
        self.experts = nn.ModuleList([Expert(in_channels, in_channels) for _ in range(view_num)])
        self.contrast_eq = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attention = Attention(in_channels) if apply_attention else nn.Identity()

    def forward(self, x_list: torch.Tensor) -> torch.Tensor:
        """
        x_list: [B, V, C, D, H, W]
        returns: [B, C, D, H, W]
        """
        if x_list.dim() != 6:
            raise ValueError(f"Expected x_list with 6 dims [B,V,C,D,H,W], got {x_list.shape}")

        B, V, C, D, H, W = x_list.shape
        if V != self.view_num:
            raise ValueError(f"Fusion block configured for V={self.view_num} but got V={V}")

        # Gate input: concat views along channel => [B, V*C, D,H,W]
        x_cat = rearrange(x_list, "b v c d h w -> b (v c) d h w")

        # Global pooling then FC => gating weights per view: [B, V]
        x_gate = nn.AdaptiveMaxPool3d((1, 1, 1))(x_cat)          # [B, V*C, 1,1,1]
        x_gate = x_gate.view(B, -1)                               # [B, V*C]
        x_gate = self.gating_network(x_gate)                      # [B, V]

        # Expert per view
        x_expert = rearrange(x_list, "b v c d h w -> v b c d h w")  # [V, B, C, D,H,W]
        out = [E(_x).unsqueeze(1) for (_x, E) in zip(x_expert, self.experts)]
        out = torch.cat(out, dim=1)                                # [B, V, C, D,H,W]

        # Apply gates
        x_gate = x_gate.view(B, V, 1, 1, 1, 1)                     # [B, V, 1,1,1,1]
        out = (out * x_gate).sum(dim=1)                            # [B, C, D,H,W]

        out = self.contrast_eq(out)
        out = self.attention(out)
        return out


# -----------------------------
# Encoder / Decoder
# -----------------------------
class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, features=(64, 128, 256)):
        super().__init__()
        f0, f1, f2 = features
        self.encoder1 = self._block(in_channels, f0)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(f0, f1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(f1, f2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = self._block(f2, f2)

    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        return enc1, enc2, enc3, bottleneck


class Decoder(nn.Module):
    def __init__(self, features=(64, 128, 256), out_channels: int = 1):
        super().__init__()
        f0, f1, f2 = features
        self.upconv3 = nn.ConvTranspose3d(f2, f2, kernel_size=2, stride=2)
        self.decoder3 = self._block(f2 + f2, f1)
        self.upconv2 = nn.ConvTranspose3d(f1, f1, kernel_size=2, stride=2)
        self.decoder2 = self._block(f1 + f1, f0)
        self.upconv1 = nn.ConvTranspose3d(f0, f0, kernel_size=2, stride=2)
        self.decoder1 = self._block(f0 + f0, f0)
        self.final_conv = nn.Conv3d(f0, out_channels, kernel_size=1)

    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc1: torch.Tensor, enc2: torch.Tensor, enc3: torch.Tensor, bottleneck: torch.Tensor):
        dec3 = self.upconv3(bottleneck)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))
        return self.final_conv(dec1)


# -----------------------------
# DoG
# -----------------------------
def difference_of_gaussians(volume_np, sigma1: float, sigma2: float):
    g1 = gaussian_filter(volume_np, sigma=sigma1)
    g2 = gaussian_filter(volume_np, sigma=sigma2)
    return g1 - g2


# -----------------------------
# Full Model (Config-driven)
# -----------------------------
class UltrasoundSR_DoGFusion(nn.Module):
    """
    Expected input:
        x: [B, V, C, D, H, W]
    Output:
        [B, out_channels, D, H, W]
    """
    def __init__(self, config: dict):
        super().__init__()

        # ---- config parsing (safe defaults) ----
        self.view_num = int(config.get("num_views", 3))
        in_channels = int(config.get("in_channels", 1))
        out_channels = int(config.get("out_channels", 1))

        features = config.get("features", (64, 128, 256))
        if isinstance(features, list):
            features = tuple(features)

        dog_cfg = config.get("dog", {})
        self.sigma1 = float(dog_cfg.get("sigma1", 0.5))
        self.sigma2 = float(dog_cfg.get("sigma2", 2.5))
        self.boost_strength = float(dog_cfg.get("boost_strength", 1.5))

        apply_attention = bool(config.get("apply_attention_bottleneck", True))

        # ---- network ----
        self.shared_encoder = SharedEncoder(in_channels=in_channels, features=features)
        self.decoder = Decoder(features=features, out_channels=out_channels)

        f0, f1, f2 = features
        self.fuse1 = Residual3DFusionBlock(in_channels=f0, view_num=self.view_num)
        self.fuse2 = Residual3DFusionBlock(in_channels=f1, view_num=self.view_num)
        self.fuse3 = Residual3DFusionBlock(in_channels=f2, view_num=self.view_num)
        self.fuse_bottleneck = Residual3DFusionBlock(in_channels=f2, view_num=self.view_num, apply_attention=apply_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, C, D, H, W = x.shape
        if V != self.view_num:
            raise ValueError(f"Model configured for {self.view_num} views, but got {V}")

        # Encode each view with shared encoder
        x_shared = rearrange(x, "b v c d h w -> (b v) c d h w")  # [(B*V), C, D,H,W]
        enc1_s, enc2_s, enc3_s, bottleneck_s = self.shared_encoder(x_shared)

        # Restore view dimension
        enc1_s = rearrange(enc1_s, "(b v) c d h w -> b v c d h w", b=B, v=V)
        enc2_s = rearrange(enc2_s, "(b v) c d h w -> b v c d h w", b=B, v=V)
        enc3_s = rearrange(enc3_s, "(b v) c d h w -> b v c d h w", b=B, v=V)
        bottleneck_s = rearrange(bottleneck_s, "(b v) c d h w -> b v c d h w", b=B, v=V)

        # Fuse across views
        enc1_f = self.fuse1(enc1_s)
        enc2_f = self.fuse2(enc2_s)
        enc3_f = self.fuse3(enc3_s)
        bottleneck_f = self.fuse_bottleneck(bottleneck_s)

        # Decode
        net_out = self.decoder(enc1_f, enc2_f, enc3_f, bottleneck_f)  # [B, out_ch, D,H,W]

        # DoG sharpening computed from input views (channel 0)
        dog_views = []
        for v in range(V):
            view = x[:, v, 0]  # [B, D, H, W]
            batch_dog = []
            for b in range(B):
                view_np = view[b].detach().cpu().to(torch.float32).numpy()
                dog = difference_of_gaussians(view_np, self.sigma1, self.sigma2)
                batch_dog.append(torch.tensor(dog, dtype=torch.float32, device=x.device))
            dog_views.append(torch.stack(batch_dog, dim=0).unsqueeze(1))  # [B,1,D,H,W]

        dog_fused = torch.cat(dog_views, dim=1).mean(dim=1)  # [B, D,H,W]

        # Final boosted output
        boosted_output = net_out + (self.boost_strength * dog_fused.unsqueeze(1))
        return boosted_output


# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":
    config = {
        "num_views": 3,
        "in_channels": 1,
        "out_channels": 1,
        "features": [64, 128, 256],
        "dog": {"sigma1": 0.5, "sigma2": 1.5, "boost_strength": 2.0},
        "apply_attention_bottleneck": True,
    }

    dummy_input = torch.randn(5, config["num_views"], config["in_channels"], 64, 64, 64)
    model = UltrasoundSR_DoGFusion(config)

    print("input shape:", dummy_input.shape)
    output = model(dummy_input)
    print("output shape:", output.shape)



