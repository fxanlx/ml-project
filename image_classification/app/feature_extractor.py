"""Image feature extraction shared by the training notebook and the Gradio app.

A CNN pretrained on ImageNet (torchvision) is used as a fixed feature extractor:
its classification head is removed, so each image becomes one feature vector
(the global-average-pooled output of the last convolutional stage).
Both the notebook and app.py import this module, so images are always
preprocessed exactly the same way at training and prediction time.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

# name -> (model builder, pretrained weights, feature dimension)
BACKBONES = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
}


class FeatureExtractor:
    def __init__(self, backbone: str = "efficientnet_b0", device: str = "cpu"):
        builder, weights, self.dim = BACKBONES[backbone]
        model = builder(weights=weights)
        if backbone.startswith("efficientnet"):
            model.classifier = nn.Identity()           # keep avgpool + flatten -> 1280-d
        else:
            model.classifier[2] = nn.Identity()        # keep LayerNorm + flatten -> 768-d
        self.model = model.eval().to(device)
        self.device = device
        self.backbone = backbone
        # Official preprocessing for these weights: resize, center-crop 224, ImageNet normalisation
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """PIL images -> array of shape (n_images, dim)."""
        feats = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(img.convert("RGB")) for img in images[i:i + batch_size]])
            feats.append(self.model(batch.to(self.device)).cpu().numpy())
        return np.concatenate(feats).astype(np.float32)

    def extract_paths(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        """Image files -> array of shape (n_images, dim). Loads one batch at a time."""
        feats = []
        for i in range(0, len(paths), batch_size):
            images = []
            for p in paths[i:i + batch_size]:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))
            feats.append(self.extract(images, batch_size))
        return np.concatenate(feats)
