import torch

from anima_calib_projfusion.model.positional_encoding import HarmonicEmbedding


def test_harmonic_embedding_output_dim():
    encoding = HarmonicEmbedding(num_harmonic_functions=6, omega_0=1.0 / 3.0)
    xy = torch.zeros(2, 10, 2)
    embedded = encoding(xy)
    assert embedded.shape == (2, 10, encoding.output_dim(2))
