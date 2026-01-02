# model_builder.py
from ssl_scan_specific.models.model import UltrasoundSR_DoGFusion
class ModelBuilder:
    def __init__(self, config):
        self.config = config

    def get_model(self):
        model_name = self.config["name"]
        model_dict = {
            "UltrasoundSR_DoGFusion": UltrasoundSR_DoGFusion,
        }
        if model_name in model_dict:
            return model_dict[model_name](self.config)
        else:
            raise ValueError(f"Model {model_name} not recognized. Choose from {list(model_dict.keys())}")
