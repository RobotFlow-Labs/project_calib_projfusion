from fastapi.testclient import TestClient

from anima_calib_projfusion.api.app import app


def test_healthz():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_calibrate_endpoint():
    client = TestClient(app)
    payload = {
        "image": [[[0.0] * 448 for _ in range(224)] for _ in range(3)],
        "point_cloud": [[0.0, 0.0, 1.0] for _ in range(8192)],
        "init_extrinsic": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "camera_info": {
            "fx": 700.0,
            "fy": 700.0,
            "cx": 600.0,
            "cy": 180.0,
            "sensor_h": 376,
            "sensor_w": 1241,
        },
        "run_iter": 3,
    }
    response = client.post("/calibrate", json=payload)
    assert response.status_code == 200
    assert response.json()["ready"] is True
