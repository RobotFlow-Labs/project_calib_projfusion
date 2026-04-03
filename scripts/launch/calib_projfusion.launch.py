from __future__ import annotations

try:
    from launch import LaunchDescription
    from launch_ros.actions import Node
except ImportError:  # pragma: no cover
    LaunchDescription = None
    Node = None


def generate_launch_description():  # pragma: no cover - ROS runtime artifact
    if LaunchDescription is None or Node is None:
        raise RuntimeError("ROS2 launch dependencies are not installed in this environment")
    return LaunchDescription(
        [
            Node(
                package="anima_calib_projfusion",
                executable="calib_projfusion_node",
                name="calib_projfusion",
                output="screen",
                parameters=[{"run_iter": 3, "checkpoint_path": ""}],
            )
        ]
    )
