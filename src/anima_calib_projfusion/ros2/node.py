from __future__ import annotations

try:
    from rclpy.node import Node
except ImportError:  # pragma: no cover

    class Node:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("rclpy is required to use the ROS2 node runtime")


from anima_calib_projfusion.inference.pipeline import iterative_calibrate
from anima_calib_projfusion.inference.service import CalibrationService
from anima_calib_projfusion.ros2.bridge import ros_to_batch


class ProjFusionCalibrationNode(Node):  # pragma: no cover - runtime integration shell
    def __init__(self) -> None:
        super().__init__("calib_projfusion")
        self.service = CalibrationService()

    def synced_callback(self, image_msg, pointcloud_msg, camera_info_msg, init_extrinsic_msg=None):
        batch = ros_to_batch(image_msg, pointcloud_msg, camera_info_msg, init_extrinsic_msg)
        pred, steps = iterative_calibrate(self.service.model, batch, run_iter=3)
        return {"pred_extrinsic": pred, "trajectory": steps}
