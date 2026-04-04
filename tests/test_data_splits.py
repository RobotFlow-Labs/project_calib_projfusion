import torch

from anima_calib_projfusion.data.perturbation import sample_perturbation


def test_sample_perturbation_shape():
    T = sample_perturbation(batch_size=4, max_deg=15.0, max_tran=0.15)
    assert T.shape == (4, 4, 4)


def test_sample_perturbation_valid_se3():
    """Perturbation matrices should be valid SE(3): det(R)=1, last row=[0,0,0,1]."""
    T = sample_perturbation(batch_size=8, max_deg=10.0, max_tran=0.5)
    # Check last row
    last_row = T[:, 3, :]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(8, -1)
    assert torch.allclose(last_row, expected, atol=1e-6)
    # Check rotation determinant ~ 1
    R = T[:, :3, :3]
    dets = torch.det(R)
    assert torch.allclose(dets, torch.ones(8), atol=1e-5)


def test_sample_perturbation_min_max():
    """With min_deg > 0, perturbation should not be identity."""
    T = sample_perturbation(batch_size=4, max_deg=10.0, max_tran=0.5, min_deg=5.0, min_tran=0.1)
    identity = torch.eye(4).unsqueeze(0).expand(4, -1, -1)
    assert not torch.allclose(T, identity, atol=1e-3)
