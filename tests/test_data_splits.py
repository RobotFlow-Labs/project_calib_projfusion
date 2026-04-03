from anima_calib_projfusion.data.perturbation import DEFAULT_PERTURBATION_RANGES, sample_uniform_perturbation
from anima_calib_projfusion.data.splits import KITTI_TEST, KITTI_TRAIN, KITTI_VAL


def test_kitti_split_lengths():
    assert len(KITTI_TRAIN) == 11
    assert len(KITTI_VAL) == 3
    assert len(KITTI_TEST) == 5
    assert KITTI_TEST == ["13", "14", "15", "16", "18"]


def test_default_perturbation_ranges():
    assert [(entry.rotation_deg, entry.translation_m) for entry in DEFAULT_PERTURBATION_RANGES] == [
        (15.0, 0.15),
        (10.0, 0.25),
        (10.0, 0.5),
    ]
    samples = sample_uniform_perturbation(4, DEFAULT_PERTURBATION_RANGES[0])
    assert samples.shape == (4, 6)
