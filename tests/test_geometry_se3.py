import torch

from anima_calib_projfusion.geometry.se3 import se3_exp, se3_inv, se3_log


def test_se3_exp_log_roundtrip():
    xi = torch.tensor([[0.1, -0.2, 0.05, 0.2, -0.1, 0.3]], dtype=torch.float64)
    transform = se3_exp(xi)
    recovered = se3_log(transform)
    assert torch.allclose(recovered, xi, atol=1e-5, rtol=1e-5)


def test_se3_inv_identity():
    T = se3_exp(torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.2]]))
    T_inv = se3_inv(T)
    product = T @ T_inv
    identity = torch.eye(4).unsqueeze(0)
    assert product.shape == (1, 4, 4)
    assert torch.allclose(product, identity, atol=1e-5)


def test_se3_exp_identity_for_zero_twist():
    xi = torch.zeros(1, 6)
    T = se3_exp(xi)
    identity = torch.eye(4).unsqueeze(0)
    assert torch.allclose(T, identity, atol=1e-6)
