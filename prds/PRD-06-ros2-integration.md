# PRD-06: ROS2 Integration

> Module: CALIB-PROJFUSION | Priority: P1
> Depends on: PRD-03, PRD-05
> Status: ⬜ Not started

## Objective
Integrate CALIB-PROJFUSION as a ROS2 node that subscribes to image, point cloud, camera info, and optional initial extrinsic topics and publishes calibrated transforms plus debug artifacts.

## Context (from paper)
The paper is built for autonomous-driving calibration scenarios where camera-LiDAR extrinsics may drift in deployment. ANIMA needs that same calibration model to be callable inside robotics pipelines.

**Paper reference**: §I, §IV-B  
**Key paper cues**: "online or in-vehicle calibration scenarios"

## Acceptance Criteria
- [ ] ROS2 node consumes `sensor_msgs/Image`, `sensor_msgs/PointCloud2`, and camera info.
- [ ] Node runs the same three-step refinement loop as the inference pipeline.
- [ ] Node publishes calibrated extrinsic transforms and optional overlay/debug messages.
- [ ] Launch file supports checkpoint path, topic remaps, device, and rate.
- [ ] Test: `uv run pytest tests/test_ros2_bridge.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_calib_projfusion/ros2/bridge.py` | ROS message conversion helpers | — | ~140 |
| `src/anima_calib_projfusion/ros2/node.py` | ROS2 calibration node | §I | ~220 |
| `scripts/launch/calib_projfusion.launch.py` | launch description | — | ~80 |
| `tests/test_ros2_bridge.py` | ROS conversion tests | — | ~100 |

## Architecture Detail (from paper)

### Inputs
- `/camera/image_raw`
- `/lidar/points`
- `/camera/camera_info`
- `/calibration/init_extrinsic` (optional)

### Outputs
- `/calibration/pred_extrinsic`
- `/tf` or `/tf_static`
- `/calibration/debug_overlay` (optional)

### Algorithm
```python
def synced_callback(image_msg, pointcloud_msg, camera_info_msg, init_extrinsic_msg=None):
    batch = ros_to_batch(image_msg, pointcloud_msg, camera_info_msg, init_extrinsic_msg)
    pred, steps = iterative_calibrate(model, batch, run_iter=3)
    publish_transform(pred)
    publish_debug(steps)
```

## Dependencies
```toml
rclpy = "*"
sensor-msgs-py = "*"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Runtime checkpoint | model | `/mnt/forge-data/models/calib_projfusion/...` | same as API/inference |

## Test Plan
```bash
uv run pytest tests/test_ros2_bridge.py -v
```

## References
- Depends on: PRD-03, PRD-05
- Feeds into: PRD-07
