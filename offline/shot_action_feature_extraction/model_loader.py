import torch
from torch import nn

from .config import SlowFastShotFeatureConfig

try:
    from pytorchvideo.models.hub import slowfast_r50
except ImportError as error:
    raise ImportError(
        "Can cai pytorchvideo truoc khi chay module nay. "
        "Vi du: pip install pytorchvideo torchvision"
    ) from error


def load_slowfast_model(
    config: SlowFastShotFeatureConfig,
    device: torch.device,
) -> nn.Module:
    use_pretrained = config.pretrained and config.model_path is None
    model = slowfast_r50(pretrained=use_pretrained)
    model.blocks[-1].proj = nn.Identity()

    if config.model_path is not None:
        checkpoint = torch.load(config.model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)

    model = model.to(device)
    model.eval()
    return model
