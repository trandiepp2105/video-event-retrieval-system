import numpy as np
import torch
from PIL import Image

from .config import PipelineConfig


class CLIPImageEmbedder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        use_cuda = config.device.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(config.device if use_cuda else "cpu")
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=str(self.config.clip_pretrained),
            device=self.device,
        )
        self.model.eval()

    @staticmethod
    def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, eps, None)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if len(images) == 0:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), self.config.batch_size):
                batch_images = images[start : start + self.config.batch_size]
                batch_tensor = torch.stack(
                    [self.preprocess(image) for image in batch_images]
                ).to(self.device)

                embeddings = self.model.encode_image(batch_tensor)
                embeddings = embeddings.float()
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                all_embeddings.append(embeddings.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return self._l2_normalize_np(embeddings).astype(np.float32)
