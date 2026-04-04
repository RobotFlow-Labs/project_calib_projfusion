from anima_calib_projfusion.api.schemas import CalibrationRequest, CameraInfoPayload
from anima_calib_projfusion.inference.service import CalibrationService


def test_service_returns_prediction_bundle():
    service = CalibrationService()
    request = CalibrationRequest(
        image=[[[0.0] * 448 for _ in range(224)] for _ in range(3)],
        point_cloud=[[0.0, 0.0, 1.0] for _ in range(8192)],
        init_extrinsic=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_info=CameraInfoPayload(
            fx=700.0,
            fy=700.0,
            cx=600.0,
            cy=180.0,
            sensor_h=376,
            sensor_w=1241,
        ),
        run_iter=3,
    )
    response = service.calibrate(request)
    assert response.ready is True
    assert len(response.pred_extrinsic) == 4
    assert len(response.trajectory) == 4
