import torch

from anima_calib_projfusion.encoders.image_dinov2 import DINOv2ImageEncoder
from anima_calib_projfusion.encoders.pointgpt import PointGPTEncoder


def test_encoder_output_shapes():
    image_encoder = DINOv2ImageEncoder()
    point_encoder = PointGPTEncoder()
    image = torch.randn(2, 3, 224, 448)
    points = torch.randn(2, 8192, 3)
    image_tokens = image_encoder(image)
    centroids, point_tokens = point_encoder(points)
    assert image_tokens.shape == (2, 512, 384)
    assert centroids.shape == (2, 128, 3)
    assert point_tokens.shape == (2, 128, 384)
