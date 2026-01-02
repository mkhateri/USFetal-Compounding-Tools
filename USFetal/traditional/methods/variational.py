import torch
import torch.nn.functional as F
from utils.logging_setup import get_logger

logger = get_logger("Variational Fusion")

# -------------------------
# Total variation loss
# -------------------------
def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv = (
        torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).sum() +
        torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).sum() +
        torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).sum()
    )
    return tv / x.numel()


# -------------------------
# 1D Gaussian kernel + 3D separable blur
# -------------------------
def gaussian_kernel1d(sigma, kernel_size, device):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()

def apply_gaussian_blur_3d(x, sigma):
    device = x.device
    k = int(2 * round(3 * sigma) + 1)
    g = gaussian_kernel1d(sigma, k, device)
    g_x = g.view(1, 1, k, 1, 1)
    g_y = g.view(1, 1, 1, k, 1)
    g_z = g.view(1, 1, 1, 1, k)
    p = k // 2
    x = F.conv3d(x, g_x, padding=(p, 0, 0), groups=1)
    x = F.conv3d(x, g_y, padding=(0, p, 0), groups=1)
    x = F.conv3d(x, g_z, padding=(0, 0, p), groups=1)
    return x

def dog_filter(x: torch.Tensor, sigma1=1.0, sigma2=2.0) -> torch.Tensor:
    return apply_gaussian_blur_3d(x, sigma1) - apply_gaussian_blur_3d(x, sigma2)

# -------------------------
# Gradient (high-pass) filter
# -------------------------
def gradient_filter(x: torch.Tensor) -> torch.Tensor:
    dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
    dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1))
    dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1))
    return torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)


# -------------------------
# Main variational fusion function
# -------------------------
def run_variational(views: torch.Tensor, config: dict, mode="standard") -> torch.Tensor:
    """
    Variational fusion using:
    - MSE Fidelity
    - Total Variation
    - Optional: DoG consistency or Gradient (high-pass) loss
    """
    logger.info(f"Running variational fusion mode: {mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    views = views.to(device)

    B, V, C, D, H, W = views.shape
    assert B == 1 and C == 1
    views_3d = views[0, :, 0]

    # Config
    alpha = config.get("alpha", 5.0)      # TV
    beta = config.get("beta", 50.1)       # Detail term
    gamma = config.get("gamma", 0.01)     # Fidelity
    lr = config.get("lr", 0.01)
    steps = config.get("steps", 100)
    sigma1 = config.get("sigma1", 1.0)
    sigma2 = config.get("sigma2", 2.0)

    logger.info(f"α={alpha}, β={beta}, γ={gamma}, steps={steps}, lr={lr}")

    # Init fused volume
    init = views_3d.mean(dim=0, keepdim=True).unsqueeze(0).to(device)
    noise = 0.005 * init.std() * torch.randn_like(init)
    fused = (init + noise).clone().detach().requires_grad_(True)

    optimizer = torch.optim.AdamW([fused], lr=lr)
    prev_loss = None

    for step in range(steps):
        optimizer.zero_grad()

        #fidelity_loss = F.mse_loss(fused.squeeze(0).repeat(V, 1, 1, 1), views_3d)
        fidelity_loss = F.l1_loss(fused.squeeze(0).repeat(V,1,1,1), views_3d)

        if mode == "dog":
            fused_feat = dog_filter(fused, sigma1, sigma2).squeeze(0).repeat(V, 1, 1, 1)
            views_feat = dog_filter(views_3d.unsqueeze(1), sigma1, sigma2).squeeze(1)
        elif mode == "gradient":
            fused_feat = gradient_filter(fused).squeeze(0).repeat(V, 1, 1, 1)
            views_feat = gradient_filter(views_3d.unsqueeze(1)).squeeze(1)
        else:
            fused_feat = views_feat = 0.0  # no detail loss

        detail_loss = F.l1_loss(fused_feat, views_feat) if mode in {"dog", "gradient"} else 0.0
        tv_val = total_variation(fused)
        total_loss = gamma * fidelity_loss + alpha * tv_val + beta * detail_loss

        total_loss.backward()
        optimizer.step()

        if step % 2 == 0 or step == steps - 1:
            delta = 0 if prev_loss is None else prev_loss - total_loss.item()
            prev_loss = total_loss.item()
            logger.info(f"Step {step:03d} | Δ={delta:.6f} | Loss={total_loss.item():.4f} | "
                        f"Fidelity={fidelity_loss.item():.4f} | Detail={detail_loss:.4f} | TV={tv_val.item():.4f}")

    return fused.detach().unsqueeze(0)  # [1, 1, D, H, W]


# -------------------------
# Registry for all modes
# -------------------------
METHOD_REGISTRY = {
    "variational[dog]": lambda views, cfg: run_variational(views, cfg, mode="dog"),
    "variational[gradient]": lambda views, cfg: run_variational(views, cfg, mode="gradient"),
}

