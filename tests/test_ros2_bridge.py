import numpy as np

from anima_calib_projfusion.ros2.bridge import ros_to_batch


def test_ros_bridge_shapes():
    image_msg = {
        "data": np.zeros((224, 448, 3), dtype=np.float32),
        "height": 224,
        "width": 448,
    }
    pointcloud_msg = {"points": np.zeros((8192, 3), dtype=np.float32)}
    camera_info_msg = {
        "fx": 700.0,
        "fy": 700.0,
        "cx": 600.0,
        "cy": 180.0,
        "sensor_h": 376,
        "sensor_w": 1241,
    }
    batch = ros_to_batch(image_msg, pointcloud_msg, camera_info_msg)
    assert batch.image.shape == (1, 3, 224, 448)
    assert batch.point_cloud.shape == (1, 8192, 3)
    assert batch.init_extrinsic.shape == (1, 4, 4)
