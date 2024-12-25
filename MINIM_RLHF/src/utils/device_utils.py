import torch
from typing import Dict

def get_device(config: Dict) -> torch.device:
    """Determine the device to use for computation."""
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def move_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move the model to the specified device."""
    model = model.to(device)
    if device.type == "cuda" and model.config["device"]["precision"] == "float16":
        model = model.half()
    return model
