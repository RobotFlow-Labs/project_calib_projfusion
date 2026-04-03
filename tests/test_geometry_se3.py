import torch

from anima_calib_projfusion.geometry.se3 import compose_transform, se3_exp, se3_log


def test_se3_exp_log_roundtrip():
    xi = torch.tensor([[0.1, -0.2, 0.05, 0.2, -0.1, 0.3]], dtype=torch.float64)
    transform = se3_exp(xi)
    recovered = se3_log(transform)
    assert torch.allclose(recovered, xi, atol=1e-5, rtol=1e-5)


def test_compose_transform_contract():
    base = torch.eye(4).unsqueeze(0)
    delta = se3_exp(torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.2]]))
    composed = compose_transform(delta, base)
    assert composed.shape == (1, 4, 4)
