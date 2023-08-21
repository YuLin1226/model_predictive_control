#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def solveAtan2Continuity(cyaw):

    for i, yaw in enumerate(cyaw):
        if yaw < 0:
            cyaw[i] = yaw + 2 * math.pi

    return cyaw


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, w=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.w = w
        
class CarViz:

    def __init__(self, car_size=[1.5, 0.6]) -> None:
        
        self.length_ = car_size[0]
        self.width_ = car_size[1]

    def plotMRS(self, x, y, yaw, truckcolor="-k"): 
    
        outline = np.array([[-self.length_/2, self.length_/2, self.length_/2, -self.length_/2, -self.length_/2],
                            [self.width_ / 2, self.width_ / 2, - self.width_ / 2, -self.width_ / 2, self.width_ / 2]])

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])

        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), truckcolor)
        
    def plotAMR(self, x, y, yaw, truckcolor="-k"): 
    
        length = 0.5

        outline = np.array([[-length/2, -length/2, +length/2, length/1.2,  +length/2, -length/2],
                            [-length/2, +length/2, +length/2, 0,           -length/2, -length/2]])

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])

        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), truckcolor)

    def vizOn(self):
        plt.show()

    def plotTrajectory(self, cx, cy, traj_format='k-', label='label'):
        plt.plot(cx, cy, traj_format, label=label)

    def plotState(self, state, state_type='center'):
        if state_type == 'center':
            self.plotMRS(state.x, state.y, state.yaw)
        elif state_type == 'amr':
            self.plotAMR(state.x, state.y, state.yaw)

    def showAnimation(self, ox, oy, cx, cy, x, y, xf, yf, xr, yr, xref, target_ind, state, state_f, state_r):

        plt.cla()
        if ox is not None:
            self.plotTrajectory(ox, oy, 'xr', 'MPC')
        self.plotTrajectory(cx, cy, 'r-', 'center_global_ref')
        self.plotTrajectory(x, y, 'bo', 'center_passed')
        self.plotTrajectory(xf, yf, 'go', 'front_passed')
        self.plotTrajectory(xr, yr, 'yo', 'rear_passed')
        self.plotTrajectory(xref[0, :], xref[1, :], 'xk', 'c_ref')
        self.plotTrajectory(xref[3, :], xref[4, :], 'xk', 'f_ref')
        self.plotTrajectory(xref[6, :], xref[7, :], 'xk', 'r_ref')
        self.plotState(state, 'center')
        self.plotState(state_f, 'amr')
        self.plotState(state_r, 'amr')
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
    
class DiffDrivedRobotModel:

    def __init__(self) -> None:
        pass

    def updateState(self, state, vx, w, dt=0.2):

        state.x = state.x + vx * math.cos(state.yaw) * dt
        state.y = state.y + vx * math.sin(state.yaw) * dt
        state.yaw = state.yaw + w * dt
        state.vx = vx
        state.vy = 0.0
        state.w = w
        return state

class GeneralBicycleModel:

    def __init__(self, nx=9, nu=4) -> None:

        self.nx_ = nx
        self.nu_ = nu

    def updateState(self, state:State, state_f:State, state_r:State):
        
        wheel_base = math.sqrt((state_f.y - state_f.y)**2 + (state_f.x - state_r.x)**2)
        state.x = (state_f.x + state_r.x) / 2
        state.y = (state_f.y + state_r.y) / 2
        state.yaw = math.atan2(state_f.y - state_r.y, state_f.x - state_r.x)
        state.vx = (state_f.vx * math.cos(state_f.yaw) + state_r.vx * math.cos(state_r.yaw)) / 2
        state.vy = (state_f.vx * math.sin(state_f.yaw) + state_r.vx * math.sin(state_r.yaw)) / 2
        state.w = 1 / wheel_base * (state_f.vx * math.sin(state_f.yaw - state.yaw) + state_r.vx * math.sin(state_r.yaw - state.yaw))
        return state

    def transformWheelCommandToRobotCommand(self, vf, vr, sf, sr, wheel_base=1):
        """
        Wheel Command: wheel speed & wheel steer
        Robot Command: Vx, Vy, W
        """
        H = np.array([
            [1, 0, 0],
            [0, 1, wheel_base/2],
            [1, 0, 0],
            [0, 1, -wheel_base/2]
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

    def getRobotModelMatrice(self, x, y, theta, x_f, y_f, theta_f, x_r, y_r, theta_r, v_f, v_r, w_f, w_r, dt=0.2):
        """
        Robot Model: x(k+1) = A x(k) + B u(k) + C
        """
        # Define L bar
        L = math.sqrt((x_f - x_r)**2 + (y_f - y_r)**2)
        # Define K bar
        K = v_f * math.sin(theta_f - theta) - v_r * math.sin(theta_r - theta)
        # Define A
        A = np.zeros((self.nx_, self.nx_))
        A[0, 0] = 1
        A[0, 5] = -dt * v_f * math.sin(theta_f) / 2
        A[0, 8] = -dt * v_r * math.sin(theta_r) / 2
        A[1, 1] = 1
        A[1, 5] = dt * v_f * math.cos(theta_f) / 2
        A[1, 8] = dt * v_r * math.cos(theta_r) / 2
        A[2, 2] = 1 + dt * (-v_f * math.cos(theta_f - theta) + v_r * math.cos(theta_r - theta)) / L
        A[2, 3] = -dt * (x_f - x_r) * ( L ** (-3) ) * K
        A[2, 4] = -dt * (y_f - y_r) * ( L ** (-3) ) * K
        A[2, 5] = dt * v_f * math.cos(theta_f - theta) / L
        A[2, 6] = dt * (x_f - x_r) * ( L ** (-3) ) * K
        A[2, 7] = dt * (y_f - y_r) * ( L ** (-3) ) * K
        A[2, 8] = dt * v_r * math.cos(theta_r - theta) / L
        A[3, 3] = 1
        A[3, 5] = -dt * v_f * math.sin(theta_f)
        A[4, 4] = 1
        A[4, 5] = dt * v_f * math.cos(theta_f)
        A[5, 5] = 1
        A[6, 5] = -dt * v_r * math.sin(theta_r)
        A[6, 6] = 1
        A[7, 5] = dt * v_r * math.cos(theta_r)
        A[7, 7] = 1
        A[8, 8] = 1
        # Define B
        B = np.zeros((self.nx_, self.nu_))
        B[0, 0] = 1 / 2 * dt * math.cos(theta_f)
        B[0, 2] = 1 / 2 * dt * math.cos(theta_r)
        B[1, 0] = 1 / 2 * dt * math.sin(theta_f)
        B[1, 2] = 1 / 2 * dt * math.sin(theta_r)
        B[2, 0] = 1 / L * dt * math.sin(theta_f - theta)
        B[2, 2] = 1 / L * dt * math.sin(theta_r - theta)
        B[3, 0] = dt * math.cos(theta_f)
        B[4, 0] = dt * math.sin(theta_f)
        B[5, 1] = dt
        B[6, 2] = dt * math.cos(theta_r)
        B[7, 2] = dt * math.sin(theta_r)
        B[8, 3] = dt
        # Define C
        C = np.zeros(self.nx_)
        C[0] = dt / 2 * (v_f * theta_f * math.sin(theta_f) + v_r * theta_r * math.sin(theta_r))
        C[1] = -dt / 2 * (v_f * theta_f * math.cos(theta_f) + v_r * theta_r * math.cos(theta_r))
        C[2] = dt / L * ((theta_r - theta) * v_r * math.cos(theta_r - theta) - (theta_f - theta) * v_f * math.cos(theta_f - theta) + K)
        C[3] = dt * v_f * theta_f * math.sin(theta_f)
        C[4] = -dt * v_f * theta_f * math.cos(theta_f)
        C[6] = dt * v_r * theta_r * math.sin(theta_r)
        C[7] = -dt * v_r * theta_r * math.cos(theta_r)
        return A, B, C

class MPC:

    def __init__(self, horizon=5, nx=9, nu=4, xy_tolerance=1.5, stop_speed=0.2, show_animation=True) -> None:
        
        self.horizon_ = horizon
        self.nx_ = nx
        self.nu_ = nu
        self.gbm_ = GeneralBicycleModel(nx=nx, nu=nu)
        self.ddrm_ = DiffDrivedRobotModel()
        self.xy_tolerance_ = xy_tolerance
        self.stop_speed_ = stop_speed

        # constraints setting - ackermann mode
        self.ackermann_rotation_speed_inc_rate_ = np.deg2rad(90)
        self.ackermann_traction_speed_inc_rate_ = 1.0
        self.ackermann_max_traction_speed_ = 0.5
        self.ackermann_max_rotation_speed_ = np.deg2rad(45)
        # constraints setting - differential mode
        self.diff_rotation_speed_inc_rate_ = np.deg2rad(90)
        self.diff_traction_speed_inc_rate_ = 0.6
        self.diff_max_traction_speed_ = 0.5
        self.diff_max_rotation_speed_ = np.deg2rad(45)
        # constraints setting - crab mode
        self.crab_rotation_speed_inc_rate_ = np.deg2rad(45)
        self.crab_traction_speed_inc_rate_ = 0.2
        self.crab_max_traction_speed_ = 0.5
        self.crab_max_rotation_speed_ = np.deg2rad(90)

        # Cost parameters
        self.R_  = np.diag([0.01, 0.01, 0.01, 0.01])
        self.Rd_ = np.diag([0.01, 0.01, 0.01, 0.01]) # Unused.
        self.Q_  = np.diag([0.0, 0.0, 0.0, 
                            0.5, 0.5, 0.1, 
                            0.5, 0.5, 0.1])
        self.Qf_ = np.diag([0.0, 0.0, 0.0, 
                            0.1, 0.1, 0.1, 
                            0.1, 0.1, 0.1])

        # Car viz
        self.viz_ = CarViz()
        self.show_animation_ = show_animation

    def doSimulationCrab(self, cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r, 
                     initial_state:State, initial_state_f:State, initial_state_r:State, 
                     mode='crab', max_time=500, dt=0.2):

        goal = [cx[-1], cy[-1]]
        state = initial_state
        state_f = initial_state_f
        state_r = initial_state_r
        
        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        vx = [state.vx]
        vy = [state.vy]
        w = [state.w]

        xf = [state_f.x]
        yf = [state_f.y]
        yawf = [state_f.yaw]

        xr = [state_r.x]
        yr = [state_r.y]
        yawr = [state_r.yaw]

        t = [0.0]
        target_ind, _ = self.getNearestIndex(state, cx, cy, cyaw, 0)
        ovf, ovr, owf, owr = None, None, None, None

        while max_time >= time:
            x0 = [state.x, state.y, state.yaw, 
                  state_f.x, state_f.y, state_f.yaw, 
                  state_r.x, state_r.y, state_r.yaw] 
            xref, target_ind = self.getReferenceTrajectory(state, cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r, 2, target_ind)
            ovf, ovr, owf, owr, ox, oy, oyaw = self.iterativeLMPC(xref, x0, ovf, ovr, owf, owr, mode)

            if ovf is not None:
                vf, vr, wf, wr = ovf[0], ovr[0], owf[0], owr[0]
                state_f = self.ddrm_.updateState(state=state_f, vx=vf, w=wf)
                state_r = self.ddrm_.updateState(state=state_r, vx=vr, w=wr)
                state = self.gbm_.updateState(state=state, state_f=state_f, state_r=state_r)
                
            time = time + dt

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            vx.append(state.vx)
            vy.append(state.vy)
            w.append(state.w)
            t.append(time)

            xf.append(state_f.x)
            yf.append(state_f.y)
            yawf.append(state_f.yaw)

            xr.append(state_r.x)
            yr.append(state_r.y)
            yawr.append(state_r.yaw)

            if self.checkGoal(state, goal, target_ind, len(cx)):
                print("Goal Reached.")
                break

            if self.show_animation_:
                self.viz_.showAnimation(ox, oy, cx, cy, x, y, xf, yf, xr, yr, xref, target_ind, state, state_f, state_r)

        # return t, x, y, yaw, vx, vy, w, state
        return t, x, y, yaw, xf, yf, yawf, xr, yr, yawr


    def doSimulationAckermann(self, cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r, 
                     initial_state:State, initial_state_f:State, initial_state_r:State, 
                     mode='ackermann', max_time=500, dt=0.2):

        goal = [cx[-1], cy[-1]]
        state = initial_state
        state_f = initial_state_f
        state_r = initial_state_r
        
        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        vx = [state.vx]
        vy = [state.vy]
        w = [state.w]

        xf = [state_f.x]
        yf = [state_f.y]
        yawf = [state_f.yaw]

        xr = [state_r.x]
        yr = [state_r.y]
        yawr = [state_r.yaw]

        t = [0.0]
        target_ind, _ = self.getNearestIndex(state, cx, cy, cyaw, 0)
        ovf, ovr, owf, owr = None, None, None, None
        cyaw = smooth_yaw(cyaw)

        while max_time >= time:
            x0 = [state.x, state.y, state.yaw, 
                  state_f.x, state_f.y, state_f.yaw, 
                  state_r.x, state_r.y, state_r.yaw] 
            xref, target_ind = self.getReferenceTrajectory(state, cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r, 2, target_ind)
            ovf, ovr, owf, owr, ox, oy, oyaw = self.iterativeLMPC(xref, x0, ovf, ovr, owf, owr, mode)

            if ovf is not None:
                vf, vr, wf, wr = ovf[0], ovr[0], owf[0], owr[0]
                state_f = self.ddrm_.updateState(state=state_f, vx=vf, w=wf)
                state_r = self.ddrm_.updateState(state=state_r, vx=vr, w=wr)
                state = self.gbm_.updateState(state=state, state_f=state_f, state_r=state_r)
                
            time = time + dt

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            vx.append(state.vx)
            vy.append(state.vy)
            w.append(state.w)
            t.append(time)

            xf.append(state_f.x)
            yf.append(state_f.y)
            yawf.append(state_f.yaw)

            xr.append(state_r.x)
            yr.append(state_r.y)
            yawr.append(state_r.yaw)

            if self.checkGoal(state, goal, target_ind, len(cx)):
                print("Goal Reached.")
                break

            if self.show_animation_:
                self.viz_.showAnimation(ox, oy, cx, cy, x, y, xf, yf, xr, yr, xref, target_ind, state, state_f, state_r)

        return t, x, y, yaw, vx, vy, w, state

    def doSimulationDiff(self, cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r, 
                     initial_state:State, initial_state_f:State, initial_state_r:State, 
                     mode='diff', max_time=500, dt=0.2):

        goal = [cx[-1], cy[-1]]
        state = initial_state
        state_f = initial_state_f
        state_r = initial_state_r
        
        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        vx = [state.vx]
        vy = [state.vy]
        w = [state.w]

        xf = [state_f.x]
        yf = [state_f.y]
        yawf = [state_f.yaw]

        xr = [state_r.x]
        yr = [state_r.y]
        yawr = [state_r.yaw]

        t = [0.0]
        target_ind, _ = self.getNearestIndex(state, cx, cy, cyaw, 0)
        ovf, ovr, owf, owr = None, None, None, None
        cyaw = smooth_yaw(cyaw)

        while max_time >= time:
            x0 = [state.x, state.y, state.yaw, 
                  state_f.x, state_f.y, state_f.yaw, 
                  state_r.x, state_r.y, state_r.yaw] 
            xref, target_ind = self.getReferenceTrajectory(state, cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r, 2, target_ind)
            ovf, ovr, owf, owr, ox, oy, oyaw = self.iterativeLMPC(xref, x0, ovf, ovr, owf, owr, mode)

            if ovf is not None:
                vf, vr, wf, wr = ovf[0], ovr[0], owf[0], owr[0]
                state_f = self.ddrm_.updateState(state=state_f, vx=vf, w=wf)
                state_r = self.ddrm_.updateState(state=state_r, vx=vr, w=wr)
                state = self.gbm_.updateState(state=state, state_f=state_f, state_r=state_r)
                
            time = time + dt

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            vx.append(state.vx)
            vy.append(state.vy)
            w.append(state.w)
            t.append(time)

            xf.append(state_f.x)
            yf.append(state_f.y)
            yawf.append(state_f.yaw)

            xr.append(state_r.x)
            yr.append(state_r.y)
            yawr.append(state_r.yaw)

            if self.checkGoal(state, goal, target_ind, len(cx)):
                print("Goal Reached.")
                break

            if self.show_animation_:
                self.viz_.showAnimation(ox, oy, cx, cy, x, y, xf, yf, xr, yr, xref, target_ind, state, state_f, state_r)

        # return t, x, y, yaw, vx, vy, w, state
        return t, x, y, yaw, xf, yf, yawf, xr, yr, yawr

    def predictMotion(self, x0, vf, vr, wf, wr, xref):
        """
        x0 = [x, y, theta, xf, yf, theta_f, xr, yr, theta_r]
        """
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[2])
        state_f = State(x=x0[3], y=x0[4], yaw=x0[5])
        state_r = State(x=x0[6], y=x0[7], yaw=x0[8])
        for (vfi, vri, wfi, wri, i) in zip(vf, vr, wf, wr, range(1, self.horizon_ + 1)):

            state_f = self.ddrm_.updateState(state=state_f, vx=vfi, w=wfi)
            state_r = self.ddrm_.updateState(state=state_r, vx=vri, w=wri)
            state = self.gbm_.updateState(state=state, state_f=state_f, state_r=state_r)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.yaw
            xbar[3, i] = state_f.x
            xbar[4, i] = state_f.y
            xbar[5, i] = state_f.yaw
            xbar[6, i] = state_r.x
            xbar[7, i] = state_r.y
            xbar[8, i] = state_r.yaw

        return xbar

    def iterativeLMPC(self, xref, x0, ovf, ovr, owf, owr, mode='ackermann', max_iter=1):
        
        ox, oy, oyaw = None, None, None
        if ovf is None or ovr is None or owf is None or owr is None:
            ovf = [0.0] * self.horizon_
            ovr = [0.0] * self.horizon_
            owf = [0.0] * self.horizon_
            owr = [0.0] * self.horizon_
        uref = np.zeros((self.nu_, self.horizon_))
        for i in range(max_iter):
            xbar = self.predictMotion(x0=x0, vf=ovf, vr=ovr, wf=owf, wr=owr, xref=xref)
            for i in range(self.horizon_):
                uref[0, i] = ovf[i]
                uref[2, i] = ovr[i]
                uref[1, i] = owf[i]
                uref[3, i] = owr[i]

                if mode == 'ackermann':
                    ovf, ovr, owf, owr, ox, oy, oyaw = self.doLMPC_Ackermann(xref, xbar, x0, uref)
                elif mode == 'diff':
                    ovf, ovr, owf, owr, ox, oy, oyaw = self.doLMPC_Differential(xref, xbar, x0, uref)
                elif mode == 'crab':
                    ovf, ovr, owf, owr, ox, oy, oyaw = self.doLMPC_Crab(xref, xbar, x0, uref)
                else:
                    print("Mode not defined. ")
            
        return ovf, ovr, owf, owr, ox, oy, oyaw

    def doLMPC_Ackermann(self, xref, xbar, x0, uref, dt=0.2):
        
        x = cvxpy.Variable((self.nx_, self.horizon_ + 1))
        u = cvxpy.Variable((self.nu_, self.horizon_))

        cost = 0.0
        constraints = []

        for t in range(self.horizon_):
            cost += cvxpy.quad_form(u[:, t], self.R_)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q_)

            A, B, C = self.gbm_.getRobotModelMatrice(
                x = xbar[0, t],
                y = xbar[1, t],
                theta = xbar[2, t],
                x_f = xbar[3, t],
                y_f = xbar[4, t],
                theta_f = xbar[5, t],
                x_r = xbar[6, t],
                y_r = xbar[7, t],
                theta_r = xbar[8, t],
                v_f = uref[0, t],
                v_r = uref[2, t],
                w_f = uref[1, t],
                w_r = uref[3, t]
                )
        
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.horizon_ - 1):
                constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= self.ackermann_traction_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= self.ackermann_traction_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= self.ackermann_rotation_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= self.ackermann_rotation_speed_inc_rate_ * dt]

            if t == 0:
                constraints += [cvxpy.abs(uref[1, t] - u[1, t]) <= self.ackermann_rotation_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(uref[3, t] - u[3, t]) <= self.ackermann_rotation_speed_inc_rate_ * dt]
    
        cost += cvxpy.quad_form(xref[:, self.horizon_] - x[:, self.horizon_], self.Qf_)


        constraints += [x[0, 0] == x0[0]]
        constraints += [x[1, 0] == x0[1]]
        constraints += [x[2, 0] == x0[2]]
        constraints += [x[3, 0] == x0[3]]
        constraints += [x[4, 0] == x0[4]]
        constraints += [x[5, 0] == x0[5]]
        constraints += [x[6, 0] == x0[6]]
        constraints += [x[7, 0] == x0[7]]
        constraints += [x[8, 0] == x0[8]]
        constraints += [cvxpy.abs(u[0, :]) <= self.ackermann_max_traction_speed_]
        constraints += [cvxpy.abs(u[2, :]) <= self.ackermann_max_traction_speed_]
        constraints += [cvxpy.abs(u[1, :]) <= self.ackermann_max_rotation_speed_]
        constraints += [cvxpy.abs(u[3, :]) <= self.ackermann_max_rotation_speed_]
        # constraints += [x[5, :] - x[2, :] == x[8, :] - x[2, :]]

        # I think w should be with different sign.
        # constraints += [u[1, :] == -u[3, :]]
        # constraints += [u[0, :] == u[2, :]]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
            oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
            oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
            ovf = np.array(u.value[0, :]).flatten()
            ovr = np.array(u.value[2, :]).flatten()
            owf = np.array(u.value[1, :]).flatten()
            owr = np.array(u.value[3, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            ox = None
            oy = None
            oyaw = None
            ovf = None
            ovr = None
            owf = None
            owr = None

        return ovf, ovr, owf, owr, ox, oy, oyaw

    def doLMPC_Differential(self, xref, xbar, x0, uref, dt=0.2):
        
        x = cvxpy.Variable((self.nx_, self.horizon_ + 1))
        u = cvxpy.Variable((self.nu_, self.horizon_))

        cost = 0.0
        constraints = []

        for t in range(self.horizon_):
            cost += cvxpy.quad_form(u[:, t], self.R_)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q_)

            A, B, C = self.gbm_.getRobotModelMatrice(
                x = xbar[0, t],
                y = xbar[1, t],
                theta = xbar[2, t],
                x_f = xbar[3, t],
                y_f = xbar[4, t],
                theta_f = xbar[5, t],
                x_r = xbar[6, t],
                y_r = xbar[7, t],
                theta_r = xbar[8, t],
                v_f = uref[0, t],
                v_r = uref[2, t],
                w_f = uref[1, t],
                w_r = uref[3, t]
                )
        
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.horizon_ - 1):
                constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= self.diff_traction_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= self.diff_traction_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= self.diff_rotation_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= self.diff_rotation_speed_inc_rate_ * dt]
    
        cost += cvxpy.quad_form(xref[:, self.horizon_] - x[:, self.horizon_], self.Qf_)

        constraints += [x[0, 0] == x0[0]]
        constraints += [x[1, 0] == x0[1]]
        constraints += [x[2, 0] == x0[2]]
        constraints += [x[3, 0] == x0[3]]
        constraints += [x[4, 0] == x0[4]]
        constraints += [x[5, 0] == x0[5]]
        constraints += [x[6, 0] == x0[6]]
        constraints += [x[7, 0] == x0[7]]
        constraints += [x[8, 0] == x0[8]]
        constraints += [cvxpy.abs(u[0, :]) <= self.diff_max_traction_speed_]
        constraints += [cvxpy.abs(u[2, :]) <= self.diff_max_traction_speed_]
        constraints += [cvxpy.abs(u[1, :]) <= self.diff_max_rotation_speed_]
        constraints += [cvxpy.abs(u[3, :]) <= self.diff_max_rotation_speed_]
        # constraints += [cvxpy.abs(x[5, :] - x[2, :] + math.pi / 2) <= np.deg2rad(30)]
        # constraints += [cvxpy.abs(x[8, :] - x[2, :] + math.pi / 2) <= np.deg2rad(30)]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
            oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
            oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
            ovf = np.array(u.value[0, :]).flatten()
            ovr = np.array(u.value[2, :]).flatten()
            owf = np.array(u.value[1, :]).flatten()
            owr = np.array(u.value[3, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            ox = None
            oy = None
            oyaw = None
            ovf = None
            ovr = None
            owf = None
            owr = None

        return ovf, ovr, owf, owr, ox, oy, oyaw

    def doLMPC_Crab(self, xref, xbar, x0, uref, dt=0.2):

        x = cvxpy.Variable((self.nx_, self.horizon_ + 1))
        u = cvxpy.Variable((self.nu_, self.horizon_))

        cost = 0.0
        constraints = []

        for t in range(self.horizon_):
            cost += cvxpy.quad_form(u[:, t], self.R_)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q_)

            A, B, C = self.gbm_.getRobotModelMatrice(
                x = xbar[0, t],
                y = xbar[1, t],
                theta = xbar[2, t],
                x_f = xbar[3, t],
                y_f = xbar[4, t],
                theta_f = xbar[5, t],
                x_r = xbar[6, t],
                y_r = xbar[7, t],
                theta_r = xbar[8, t],
                v_f = uref[0, t],
                v_r = uref[2, t],
                w_f = uref[1, t],
                w_r = uref[3, t]
                )
        
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.horizon_ - 1):
                constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= self.crab_traction_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= self.crab_traction_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= self.crab_rotation_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= self.crab_rotation_speed_inc_rate_ * dt]
    
        cost += cvxpy.quad_form(xref[:, self.horizon_] - x[:, self.horizon_], self.Qf_)

        constraints += [x[0, 0] == x0[0]]
        constraints += [x[1, 0] == x0[1]]
        constraints += [x[2, 0] == x0[2]]
        constraints += [x[3, 0] == x0[3]]
        constraints += [x[4, 0] == x0[4]]
        constraints += [x[5, 0] == x0[5]]
        constraints += [x[6, 0] == x0[6]]
        constraints += [x[7, 0] == x0[7]]
        constraints += [x[8, 0] == x0[8]]
        constraints += [cvxpy.abs(u[0, :]) <= self.crab_max_traction_speed_]
        constraints += [cvxpy.abs(u[2, :]) <= self.crab_max_traction_speed_]
        constraints += [cvxpy.abs(u[1, :]) <= self.crab_max_rotation_speed_]
        constraints += [cvxpy.abs(u[3, :]) <= self.crab_max_rotation_speed_]
        constraints += [x[5, :] == x[8, :]]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
            oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
            oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
            ovf = np.array(u.value[0, :]).flatten()
            ovr = np.array(u.value[2, :]).flatten()
            owf = np.array(u.value[1, :]).flatten()
            owr = np.array(u.value[3, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            ox = None
            oy = None
            oyaw = None
            ovf = None
            ovr = None
            owf = None
            owr = None

        return ovf, ovr, owf, owr, ox, oy, oyaw

    def getNearestIndex(self, state, cx, cy, cyaw, pind, n_ind_search=10):

        dx = [state.x - icx for icx in cx[pind:(pind + n_ind_search)]]
        dy = [state.y - icy for icy in cy[pind:(pind + n_ind_search)]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        mind = min(d)
        ind = d.index(mind) + pind
        mind = math.sqrt(mind)
        return ind, mind

    def getReferenceTrajectory(self, state, cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r, n_search_ind, pind, wheel_base=1):
        
        xref = np.zeros((self.nx_, self.horizon_ + 1))
        ind, _ = self.getNearestIndex(state, cx, cy, cyaw, pind)
        if pind >= ind:
            ind = pind
        feasible_cx = cx[ind: ind + n_search_ind]
        feasible_cy = cy[ind: ind + n_search_ind]
        feasible_cyaw = cyaw[ind: ind + n_search_ind]
        feasible_cx_f = cx_f[ind: ind + n_search_ind]
        feasible_cy_f = cy_f[ind: ind + n_search_ind]
        feasible_cyaw_f = cyaw_f[ind: ind + n_search_ind]
        feasible_cx_r = cx_r[ind: ind + n_search_ind]
        feasible_cy_r = cy_r[ind: ind + n_search_ind]
        feasible_cyaw_r = cyaw_r[ind: ind + n_search_ind]
        ncourse = len(feasible_cx)
        for i in range(self.horizon_ + 1):
            if i < ncourse:
                xref[0, i] = feasible_cx[i]
                xref[1, i] = feasible_cy[i]
                xref[2, i] = feasible_cyaw[i]
                xref[3, i] = feasible_cx_f[i]
                xref[4, i] = feasible_cy_f[i]
                xref[5, i] = feasible_cyaw_f[i]
                xref[6, i] = feasible_cx_r[i]
                xref[7, i] = feasible_cy_r[i]
                xref[8, i] = feasible_cyaw_r[i]
            else:
                xref[0, i] = feasible_cx[ncourse - 1]
                xref[1, i] = feasible_cy[ncourse - 1]
                xref[2, i] = feasible_cyaw[ncourse - 1]
                xref[3, i] = feasible_cx_f[ncourse - 1]
                xref[4, i] = feasible_cy_f[ncourse - 1]
                xref[5, i] = feasible_cyaw_f[ncourse - 1]
                xref[6, i] = feasible_cx_r[ncourse - 1]
                xref[7, i] = feasible_cy_r[ncourse - 1]
                xref[8, i] = feasible_cyaw_r[ncourse - 1]
        return xref, ind
    
    def checkGoal(self, state, goal, tind, nind):

        dx = state.x - goal[0]
        dy = state.y - goal[1]
        d = math.hypot(dx, dy)
        isGoalReached = (d <= self.xy_tolerance_)
        if abs(tind - nind) >= 5:
            isGoalReached = False
        isStopped = (abs(state.vx) <= self.stop_speed_)
        if isGoalReached and isStopped:
            return True
        return False

class TrajectoryGenerator:

    def __init__(self) -> None:
        pass

    def makeEightShapeTrajectoryWithCurvature(self, size=10, n=121):
        x, y, yaw = [], [], []
        curvature = []
        for i in range(n):
            t = i / ((n - 1) / 2)
            ptx = 0.8 * math.sin(2 * math.pi * t) * size
            pty = math.sin(1 * math.pi * t) * size

            dx = 0.8 * math.cos(2 * math.pi * t) * (2 * math.pi) * size
            dy = math.cos(1 * math.pi * t) * (1 * math.pi) * size
            ptyaw = math.atan2(dy, dx)
            x.append(ptx)
            y.append(pty)
            yaw.append(ptyaw)

            ddx = 0.8 * math.sin(2 * math.pi * t) * (-1) * (2 * math.pi) * size
            ddy = math.cos(1 * math.pi * t) * (-1) * (1 * math.pi) * size
            k = (dx * ddy - ddx * dy) / ( (dx**2 + dy**2)**(1.5) )
            curvature.append(k)
        
        return x ,y, yaw, curvature

class CrabMotionTrajectoryGenerator(TrajectoryGenerator):

    def makeEightShapeTrajectory(self, wheel_base=1, size=10, n=121):

        cx, cy, cyaw, curvature = self.makeEightShapeTrajectoryWithCurvature(size=size, n=n)
        cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r = self.getFrontAndRearTrajectories(cx=cx, cy=cy, cyaw=cyaw, curvature=curvature, wheel_base=wheel_base)
        return cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r

    def getFrontAndRearTrajectories(self, cx, cy, cyaw, curvature, wheel_base):

        cx_f, cy_f, cyaw_f = [], [], []
        cx_r, cy_r, cyaw_r = [], [], []

        for x, y, yaw, k in zip(cx, cy, cyaw, curvature):

            r = 1 / k
            yaw_f = yaw 
            yaw_r = yaw 
            
            x_f = x + math.cos(cyaw[0]) * wheel_base / 2
            y_f = y + math.sin(cyaw[0]) * wheel_base / 2

            x_r = x - math.cos(cyaw[0]) * wheel_base / 2
            y_r = y - math.sin(cyaw[0]) * wheel_base / 2

            cx_f.append(x_f)
            cy_f.append(y_f)
            cyaw_f.append(yaw_f)
            cx_r.append(x_r)
            cy_r.append(y_r)
            cyaw_r.append(yaw_r)

        return cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r

class AckermannMotionTrajectoryGenerator(TrajectoryGenerator):

    def makeEightShapeTrajectory(self, wheel_base=1, size=10, n=121):

        cx, cy, cyaw, curvature = self.makeEightShapeTrajectoryWithCurvature(size=size, n=n)
        cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r = self.getFrontAndRearTrajectories(cx=cx, cy=cy, cyaw=cyaw, curvature=curvature, wheel_base=wheel_base)
        return cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r

    def getFrontAndRearTrajectories(self, cx, cy, cyaw, curvature, wheel_base):

        cx_f, cy_f, cyaw_f = [], [], []
        cx_r, cy_r, cyaw_r = [], [], []

        for x, y, yaw, k in zip(cx, cy, cyaw, curvature):

            r = 1 / k
            yaw_f = yaw - math.atan(wheel_base / 2 / r)
            yaw_r = yaw + math.atan(wheel_base / 2 / r)
            
            x_f = x + math.cos(cyaw[0]) * wheel_base / 2
            y_f = y + math.sin(cyaw[0]) * wheel_base / 2

            x_r = x - math.cos(cyaw[0]) * wheel_base / 2
            y_r = y - math.sin(cyaw[0]) * wheel_base / 2

            cx_f.append(x_f)
            cy_f.append(y_f)
            cyaw_f.append(yaw_f)
            cx_r.append(x_r)
            cy_r.append(y_r)
            cyaw_r.append(yaw_r)

        return cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r

class DifferentialMotionTrajectoryGenerator(TrajectoryGenerator):

    def makeEightShapeTrajectory(self, wheel_base=1, size=10, n=121):

        cx, cy, cyaw, curvature = self.makeEightShapeTrajectoryWithCurvature(size=size, n=n)
        # Modify cyaw to make orientation perpendicular to the path.
        for i, yaw in enumerate(cyaw):
            cyaw[i] = yaw + math.pi / 2
        cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r = self.getFrontAndRearTrajectories(cx=cx, cy=cy, cyaw=cyaw, curvature=curvature, wheel_base=wheel_base)
        return cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r

    def getFrontAndRearTrajectories(self, cx, cy, cyaw, curvature, wheel_base):

        cx_f, cy_f, cyaw_f = [], [], []
        cx_r, cy_r, cyaw_r = [], [], []

        for x, y, yaw, k in zip(cx, cy, cyaw, curvature):

            r = 1 / k
            yaw_f = yaw - math.pi / 2
            yaw_r = yaw - math.pi / 2
            
            x_f = x + math.cos(yaw) * wheel_base / 2
            y_f = y + math.sin(yaw) * wheel_base / 2

            x_r = x - math.cos(yaw) * wheel_base / 2
            y_r = y - math.sin(yaw) * wheel_base / 2

            cx_f.append(x_f)
            cy_f.append(y_f)
            cyaw_f.append(yaw_f)
            cx_r.append(x_r)
            cy_r.append(y_r)
            cyaw_r.append(yaw_r)

        return cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r


# --------------------- Main Function Section --------------------- 


def main1():
    print(__file__ + " start... MRS move in crab mode.")

    gbm_length = 1
    tg = CrabMotionTrajectoryGenerator()
    cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r = tg.makeEightShapeTrajectory(wheel_base=gbm_length)

    cyaw = solveAtan2Continuity(cyaw)
    cyaw_f = solveAtan2Continuity(cyaw_f)
    cyaw_r = solveAtan2Continuity(cyaw_r)

    # Change all cyaw to cyaw[0]
    for i in range(len(cyaw)):
        cyaw[i] = cyaw[0]

    xc, yc, yawc = cx[0], cy[0], cyaw[0]
    xf, yf, yawf = cx_f[0], cy_f[0], cyaw_f[0]
    xr, yr, yawr = cx_r[0], cy_r[0], cyaw_r[0]
    initial_state = State(x=xc, y=yc, yaw=yawc)
    initial_state_f = State(x=xf, y=yf, yaw=yawf)
    initial_state_r = State(x=xr, y=yr, yaw=yawr)

    # cx.pop(0)
    # cy.pop(0)
    # cyaw.pop(0)

    mpc = MPC()

    # Cost parameters
    mpc.R_  = np.diag([0.01, 0.01, 0.01, 0.01])
    mpc.Rd_ = np.diag([0.01, 0.01, 0.01, 0.01]) # Unused.
    mpc.Q_  = np.diag([0.5, 0.5, 0.0, 
                        0.1, 0.1, 0.01, 
                        0.1, 0.1, 0.01])
    mpc.Qf_ = np.diag([0.5, 0.5, 0.0, 
                        0.1, 0.1, 0.01, 
                        0.1, 0.1, 0.01])

    t, xc, yc, yawc, xf, yf, yawf, xr, yr, yawr = mpc.doSimulationCrab(
        cx, cy, cyaw, 
        cx_f, cy_f, cyaw_f, 
        cx_r, cy_r, cyaw_r, 
        initial_state, initial_state_f, initial_state_r)
    
    dist = []
    dyawf = []
    dyawr = []
    for i in range(len(t)):
        dist.append(math.sqrt((xf[i] - xr[i])**2 + (yf[i] - yr[i])**2))
        dyawf.append(np.rad2deg(abs(yawf[i] - yawc[i])))
        dyawr.append(np.rad2deg(abs(yawr[i] - yawc[i])))
    
    plt.figure(2)
    plt.plot(t, dist, 'r-')
    plt.figure(3)
    plt.plot(t, dyawf, 'g-')
    plt.plot(t, dyawr, 'b-')
    plt.show()

def main2():
    print(__file__ + " start... MRS move in ackermann mode.")

    gbm_length = 1
    tg = AckermannMotionTrajectoryGenerator()
    cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r = tg.makeEightShapeTrajectory(wheel_base=gbm_length)

    # Change all cyaw to cyaw[0]
    for i in range(len(cyaw)):
        cyaw[i] = cyaw[0]

    xc, yc, yawc = cx[0], cy[0], cyaw[0]
    xf, yf, yawf = cx_f[0], cy_f[0], cyaw_f[0]
    xr, yr, yawr = cx_r[0], cy_r[0], cyaw_r[0]
    initial_state = State(x=xc, y=yc, yaw=yawc)
    initial_state_f = State(x=xf, y=yf, yaw=yawf)
    initial_state_r = State(x=xr, y=yr, yaw=yawr)

    # cx.pop(0)
    # cy.pop(0)
    # cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulationAckermann(
        cx, cy, cyaw, 
        cx_f, cy_f, cyaw_f, 
        cx_r, cy_r, cyaw_r, 
        initial_state, initial_state_f, initial_state_r)

def main3():
    print(__file__ + " start... MRS move in differential mode.")

    gbm_length = 1
    tg = DifferentialMotionTrajectoryGenerator()
    cx, cy, cyaw, cx_f, cy_f, cyaw_f, cx_r, cy_r, cyaw_r = tg.makeEightShapeTrajectory(wheel_base=gbm_length, n=481)

    cyaw = solveAtan2Continuity(cyaw)
    cyaw_f = solveAtan2Continuity(cyaw_f)
    cyaw_r = solveAtan2Continuity(cyaw_r)

    xc, yc, yawc = cx[0], cy[0], cyaw[0]
    xf, yf, yawf = cx_f[0], cy_f[0], cyaw_f[0]
    xr, yr, yawr = cx_r[0], cy_r[0], cyaw_r[0]
    initial_state = State(x=xc, y=yc, yaw=yawc)
    initial_state_f = State(x=xf, y=yf, yaw=yawf)
    initial_state_r = State(x=xr, y=yr, yaw=yawr)

    mpc = MPC()
    # Cost parameters
    mpc.R_  = np.diag([0.01, 0.01, 0.01, 0.01])
    mpc.Rd_ = np.diag([0.01, 0.01, 0.01, 0.01]) # Unused.
    mpc.Q_  = np.diag([0.0, 0.0, 0.0, 
                        0.5, 0.5, 0.1, 
                        0.5, 0.5, 0.1])
    mpc.Qf_ = np.diag([0.0, 0.0, 0.0, 
                        0.1, 0.1, 0.1, 
                        0.1, 0.1, 0.1])

    t, xc, yc, yawc, xf, yf, yawf, xr, yr, yawr = mpc.doSimulationDiff(
        cx, cy, cyaw, 
        cx_f, cy_f, cyaw_f, 
        cx_r, cy_r, cyaw_r, 
        initial_state, initial_state_f, initial_state_r)
    
    dist = []
    dyawf = []
    dyawr = []
    for i in range(len(t)):
        dist.append(math.sqrt((xf[i] - xr[i])**2 + (yf[i] - yr[i])**2))
        dyawf.append(np.rad2deg(abs(yawf[i] - yawc[i])))
        dyawr.append(np.rad2deg(abs(yawr[i] - yawc[i])))
    
    plt.figure(2)
    plt.plot(t, dist, 'r-')
    plt.figure(3)
    plt.plot(t, dyawf, 'g-')
    plt.plot(t, dyawr, 'b-')
    plt.show()
    
if __name__ == '__main__':
    
    # main1() # Crab          Mode With 8-shaped Path.

    # main2() # Ackermann     Mode With 8-shaped Path.
    
    main3() # Differential  Mode With 8-shaped Path.
