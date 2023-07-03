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


# Development

State Update Function:

$$
\begin{bmatrix}
    x(k+1) \\ y(k+1) \\ \theta(k+1)
\end{bmatrix}
=
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
    x(k) \\ y(k) \\ \theta(k)
\end{bmatrix}
+
\begin{bmatrix}
    \cos{\theta(k)} & -\sin{\theta(k)} & 0 \\
    \sin{\theta(k)} & \cos{\theta(k)} & 0 \\
    0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
    V_x \\ V_y \\ \omega
\end{bmatrix}
$$

Double S/D Kinematics:

$$
H = 
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & \frac{WB}{2} \\
    1 & 0 & 0 \\
    0 & 1 & -\frac{WB}{2}
\end{bmatrix}
$$
$$
V_o = (H^TH)^{-1}H^T V_i
$$

