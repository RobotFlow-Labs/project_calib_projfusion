# CALIB-PROJFUSION PRD Suite

This folder contains the standard 7-PRD ANIMA build plan for the ProjFusion paper.

## Order
1. [PRD-01: Foundation & Config](PRD-01-foundation.md)
2. [PRD-02: Core Model](PRD-02-core-model.md)
3. [PRD-03: Inference](PRD-03-inference.md)
4. [PRD-04: Evaluation](PRD-04-evaluation.md)
5. [PRD-05: API & Docker](PRD-05-api-docker.md)
6. [PRD-06: ROS2 Integration](PRD-06-ros2-integration.md)
7. [PRD-07: Production](PRD-07-production.md)

## Notes
- PRDs are paper-faithful first and ANIMA-integrated second.
- The released repo in `repositories/ProjFusion` is treated as the executable truth when the paper and implementation diverge.
- Every PRD has a corresponding task set in `tasks/`.
