import math
import numpy as np

class GeneralBicycleModel:

    def __init__(self, wheel_base=1, nx=3, nu=4) -> None:
        self.wheel_base_ = wheel_base
        self.nx_ = nx
        self.nu_ = nu

    def updateState(self, state, vx, vy, w, dt=0.2):
        
        state.x = state.x + vx * math.cos(state.yaw) * dt - vy * math.sin(state.yaw) * dt
        state.y = state.y + vx * math.sin(state.yaw) * dt + vy * math.cos(state.yaw) * dt
        state.yaw = state.yaw + w * dt
        state.vx = vx
        state.vy = vy
        state.w = w
        return state

    def transformWheelCommandToRobotCommand(self, vf, vr, sf, sr):
        """
        Wheel Command: wheel speed & wheel steer
        Robot Command: Vx, Vy, W
        """
        H = np.array([
            [1, 0, 0],
            [0, 1, self.wheel_base_/2],
            [1, 0, 0],
            [0, 1, -self.wheel_base_/2]
        ])
        km = np.linalg.inv((H.transpose() @ H)) @ H.transpose()
        v1x = math.cos(sf) * vf
        v1y = math.sin(sf) * vf
        v2x = math.cos(sr) * vr
        v2y = math.sin(sr) * vr
        vo = np.array([
            [v1x],
            [v1y],
            [v2x],
            [v2y]
        ])
        vi = km @ vo
        vx = float(vi[0])
        vy = float(vi[1])
        w = float(vi[2])
        return vx, vy, w

    def getRobotModelMatrice(self, theta, v_f, delta_f, v_r, delta_r, dt=0.2):
        """
        Robot Model: x(k+1) = A x(k) + B u(k) + C
        """
        # Define A
        A = np.zeros((self.nx_, self.nx_))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[0, 2] = -0.5*dt*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.sin(theta) -0.5*dt*(v_f*math.sin(delta_f)+v_r*math.sin(delta_r))*math.cos(theta)
        A[1, 2] = 0.5*dt*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.cos(theta) -0.5*dt*(v_f*math.sin(delta_f)+v_r*math.sin(delta_r))*math.sin(theta)
        # Define B
        B = np.zeros((self.nx_, self.nu_))
        B[0, 0] =  1 / 2 * dt * (math.cos(delta_f) * math.cos(theta) - math.sin(delta_f) * math.sin(theta))
        B[0, 1] = -1 / 2 * dt * (math.sin(delta_f) * math.cos(theta) + math.cos(delta_f) * math.sin(theta)) * v_f
        B[0, 2] =  1 / 2 * dt * (math.cos(delta_r) * math.cos(theta) - math.sin(delta_r) * math.sin(theta))
        B[0, 3] = -1 / 2 * dt * (math.sin(delta_r) * math.cos(theta) + math.cos(delta_r) * math.sin(theta)) * v_r
        B[1, 0] =  1 / 2 * dt * (math.cos(delta_f) * math.sin(theta) + math.sin(delta_f) * math.cos(theta))
        B[1, 1] = -1 / 2 * dt * (math.sin(delta_f) * math.sin(theta) - math.cos(delta_f) * math.cos(theta)) * v_f
        B[1, 2] =  1 / 2 * dt * (math.cos(delta_r) * math.sin(theta) + math.sin(delta_r) * math.cos(theta))
        B[1, 3] = -1 / 2 * dt * (math.sin(delta_r) * math.cos(theta) - math.cos(delta_r) * math.sin(theta)) * v_r
        B[2, 0] =  1 / self.wheel_base_ * dt * math.sin(delta_r)
        B[2, 1] =  1 / self.wheel_base_ * dt * math.cos(delta_f) * v_f 
        B[2, 2] = -1 / self.wheel_base_ * dt * math.sin(delta_r)
        B[2, 3] = -1 / self.wheel_base_ * dt * math.cos(delta_r) * v_r
        # Define C
        C = np.zeros(self.nx_)
        C[0] = 1 / 2 * dt * (v_f * math.cos(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.cos(delta_r) * math.sin(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.cos(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.cos(theta) * (delta_r + theta))
        C[1] = 1 / 2 * dt * (-v_f * math.cos(delta_f) * math.cos(theta) * (delta_f + theta) - v_r * math.cos(delta_r) * math.cos(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.sin(theta) * (delta_r + theta))
        C[2] = 1 / self.wheel_base_ * dt * (-math.cos(delta_f) * delta_f * v_f + math.cos(delta_r) * delta_r * v_r)
        
        return A, B, C