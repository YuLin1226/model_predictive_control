# Requirement

This controller is for Double S/D robots.

### Pre-defined Path

Nodes:
- Pose: $x$, $y$, $\theta$
- Twist: $v_x$, $v_y$, $\omega$
- Wheel Info: front steer, front speed, rear steer, rear speed
- Time: t
- Mode: Crab, Ackermann, Differential

### Measurement

Using SLAM & TF in ROS1 to get robot states.

State:
- Pose: $x$, $y$, $\theta$
- Twist: $v_x$, $v_y$, $\omega$
- Wheel Info: front steer, front speed, rear steer, rear speed


# Notes

x_init: represent "Current State", this state should be updated from TF.

