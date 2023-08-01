#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import pathlib
import csv
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import cubic_spline_planner


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

    def plotCar(self, x, y, yaw, truckcolor="-k"): 
    
        outline = np.array([[-self.length_/2, self.length_/2, self.length_/2, -self.length_/2, -self.length_/2],
                            [self.width_ / 2, self.width_ / 2, - self.width_ / 2, -self.width_ / 2, self.width_ / 2]])

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])

        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), truckcolor)
        
    def vizOn(self):
        plt.show()

    def plotTrajectory(self, cx, cy):
        plt.plot(cx, cy, "k-")


    def showAnimation(self, ox, oy, cx, cy, x, y, xref, target_ind, state):

        # plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if ox is not None:
            plt.plot(ox, oy, "xr", label="MPC")
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(x, y, "ob", label="trajectory")
        plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
        plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        # self.plotCar(state.x, state.y, state.yaw)
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
    
    def simulateWheelCommandFromDiffDriveRobotCommand(self, vf, vr, theta_f, theta_r, theta_system):

        sf = theta_f - theta_system
        sr = theta_r - theta_system
        return vf, vr, sf, sr


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

class MPC:

    def __init__(self, horizon=5, nx=3, nu=4, xy_tolerance=1.5, stop_speed=0.2, show_animation=True) -> None:
        
        self.horizon_ = horizon
        self.nx_ = nx
        self.nu_ = nu
        self.gbm_ = GeneralBicycleModel(wheel_base=1, nx=nx, nu=nu)
        self.ddrm_ = DiffDrivedRobotModel()
        self.xy_tolerance_ = xy_tolerance
        self.stop_speed_ = stop_speed

        # constraints setting - ackermann mode
        self.ackermann_steer_inc_rate_ = np.deg2rad(45)
        self.ackermann_speed_inc_rate_ = 0.2
        self.ackermann_max_speed_ = 0.47
        self.ackermann_max_steer_ = np.deg2rad(45)
        # constraints setting - differential mode
        self.differential_speed_inc_rate_ = 0.2
        self.differential_fixed_steer_ = np.deg2rad(90)
        self.differential_max_speed_ = 0.47
        # constraints setting - crab mode
        self.crab_steer_inc_rate_ = np.deg2rad(45)
        self.crab_speed_inc_rate_ = 0.2
        self.crab_max_speed_ = 0.5
        self.crab_max_steer_ = np.deg2rad(45)

        # Cost parameters
        self.R_  = np.diag([0.01, 0.01, 0.01, 0.01])
        self.Rd_ = np.diag([0.01, 0.01, 0.01, 0.01]) # Unused.
        self.Q_  = np.diag([1.0, 1.0, 0.5])
        self.Qf_ = np.diag([1.0, 1.0, 0.5])

        # Car viz
        self.viz_ = CarViz()
        self.show_animation_ = show_animation

    def doSimulation(self, cx, cy, cyaw, initial_state, mode='ackermann', max_time=500, dt=0.2):

        goal = [cx[-1], cy[-1]]
        state = initial_state
        # initial yaw compensation
        if state.yaw - cyaw[0] >= math.pi:
            state.yaw -= math.pi * 2.0
        elif state.yaw - cyaw[0] <= -math.pi:
            state.yaw += math.pi * 2.0

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        vx = [state.vx]
        vy = [state.vy]
        w = [state.w]

        t = [0.0]
        target_ind, _ = self.getNearestIndex(state, cx, cy, cyaw, 0)
        ovf, ovr, osf, osr = None, None, None, None
        cyaw = smooth_yaw(cyaw)

        while max_time >= time:

            x0 = [state.x, state.y, state.yaw] 
            xref, target_ind = self.getReferenceTrajectory(state, cx, cy, cyaw, 2, target_ind)
            ovf, ovr, osf, osr, ox, oy, oyaw = self.iterativeLMPC(xref, x0, ovf, ovr, osf, osr, mode)

            if ovf is not None:
                vfi, vri, sfi, sri = ovf[0], ovr[0], osf[0], osr[0]
                vxi, vyi, wi = self.gbm_.transformWheelCommandToRobotCommand(vfi, vri, sfi, sri)
                state = self.gbm_.updateState(state, vxi, vyi, wi)
                
            time = time + dt

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            vx.append(state.vx)
            vy.append(state.vy)
            w.append(state.w)
            t.append(time)

            if self.checkGoal(state, goal, target_ind, len(cx)):
                print("Goal Reached.")
                break

            if self.show_animation_:
                self.viz_.showAnimation(ox, oy, cx, cy, x, y, xref, target_ind, state)

        return t, x, y, yaw, vx, vy, w, state

    def doSimulationWithAllMotionModes(self, idx_group, cx, cy, cyaw, cmode, initial_state, max_time=500, dt=0.2):
        
        self.viz_.plotTrajectory(cx, cy)

        current_state = initial_state
        for idx in idx_group:
            lx = cx[idx[0] : idx[1] + 1]
            ly = cy[idx[0] : idx[1] + 1]
            lyaw = cyaw[idx[0] : idx[1] + 1]
            lmode = cmode[idx[0]]
            t, x, y, yaw, vx, vy, w, state = self.doSimulation(lx, ly, lyaw, current_state, lmode)
            current_state = state


    
    def predictMotion(self, x0, vf, vr, wf, wr, xref):
        """
        x0 = [x, y, theta, [xf, yf, theta_f], [xr, yr, theta_r]]
        """
        xbar = xref * 0.0
        xbar[0, 0] = x0[0]
        xbar[1, 0] = x0[1]
        xbar[2, 0] = x0[2]

        state = State(x=x0[0], y=x0[1], yaw=x0[2])
        state_f = State(x=x0[3][0], y=x0[3][1], yaw=x0[3][2])
        state_r = State(x=x0[4][0], y=x0[4][1], yaw=x0[4][2])
        for (vfi, vri, wfi, wri, i) in zip(vf, vr, wf, wr, range(1, self.horizon_ + 1)):

            vfi, vri, sfi, sri = self.gbm_.simulateWheelCommandFromDiffDriveRobotCommand(vfi, vri, state_f.yaw, state_r.yaw, state.yaw)
            vx, vy, w = self.gbm_.transformWheelCommandToRobotCommand(vfi, vri, sfi, sri)
            state_f = self.ddrm_.updateState(state_f, vfi, wfi)
            state_r = self.ddrm_.updateState(state_r, vri, wri)
            state = self.gbm_.updateState(state, vx, vy, w)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.yaw

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
            xbar = self.predictMotion(x0, ovf, ovr, owf, owr, xref)
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
                theta=xbar[2, t],
                v_f=uref[0, t],
                v_r=uref[2, t],
                delta_f=uref[1, t],
                delta_r=uref[3, t])
        
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.horizon_ - 1):
                constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= self.ackermann_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= self.ackermann_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= self.ackermann_steer_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= self.ackermann_steer_inc_rate_ * dt]

            if t == 0:
                constraints += [cvxpy.abs(uref[1, t] - u[1, t]) <= self.ackermann_steer_inc_rate_ * dt]
                constraints += [cvxpy.abs(uref[3, t] - u[3, t]) <= self.ackermann_steer_inc_rate_ * dt]


        cost += cvxpy.quad_form(xref[:, self.horizon_] - x[:, self.horizon_], self.Qf_)

        constraints += [x[:, 0] == x0]
        constraints += [cvxpy.abs(u[0, :]) <= self.ackermann_max_speed_]
        constraints += [cvxpy.abs(u[2, :]) <= self.ackermann_max_speed_]
        constraints += [cvxpy.abs(u[1, :]) <= self.ackermann_max_steer_]
        constraints += [cvxpy.abs(u[3, :]) <= self.ackermann_max_steer_]
        constraints += [u[1, :] == -u[3, :]]
        constraints += [u[0, :] == u[2, :]]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
            oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
            oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
            ovf = np.array(u.value[0, :]).flatten()
            ovr = np.array(u.value[2, :]).flatten()
            osf = np.array(u.value[1, :]).flatten()
            osr = np.array(u.value[3, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            ox = None
            oy = None
            oyaw = None
            ovf = None
            ovr = None
            osf = None
            osr = None

        return ovf, ovr, osf, osr, ox, oy, oyaw

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
                theta=xbar[2, t],
                v_f=uref[0, t],
                v_r=uref[2, t],
                delta_f=uref[1, t],
                delta_r=uref[3, t])
        
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.horizon_ - 1):
                constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= self.differential_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= self.differential_speed_inc_rate_ * dt]

        cost += cvxpy.quad_form(xref[:, self.horizon_] - x[:, self.horizon_], self.Qf_)

        constraints += [x[:, 0] == x0]
        constraints += [cvxpy.abs(u[0, :]) <= self.differential_max_speed_]
        constraints += [cvxpy.abs(u[2, :]) <= self.differential_max_speed_]
        constraints += [u[1, :] == self.differential_fixed_steer_]
        constraints += [u[3, :] == self.differential_fixed_steer_]
        

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
            oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
            oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
            ovf = np.array(u.value[0, :]).flatten()
            ovr = np.array(u.value[2, :]).flatten()
            osf = np.array(u.value[1, :]).flatten()
            osr = np.array(u.value[3, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            ox = None
            oy = None
            oyaw = None
            ovf = None
            ovr = None
            osf = None
            osr = None

        return ovf, ovr, osf, osr, ox, oy, oyaw

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
                theta=xbar[2, t],
                v_f=uref[0, t],
                v_r=uref[2, t],
                delta_f=uref[1, t],
                delta_r=uref[3, t])
        
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.horizon_ - 1):
                constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= self.crab_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= self.crab_speed_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= self.crab_steer_inc_rate_ * dt]
                constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= self.crab_steer_inc_rate_ * dt]

            if t == 0:
                constraints += [cvxpy.abs(uref[1, t] - u[1, t]) <= self.crab_steer_inc_rate_ * dt]
                constraints += [cvxpy.abs(uref[3, t] - u[3, t]) <= self.crab_steer_inc_rate_ * dt]


        cost += cvxpy.quad_form(xref[:, self.horizon_] - x[:, self.horizon_], self.Qf_)

        constraints += [x[:, 0] == x0]
        constraints += [cvxpy.abs(u[0, :]) <= self.crab_max_speed_]
        constraints += [cvxpy.abs(u[2, :]) <= self.crab_max_speed_]
        constraints += [cvxpy.abs(u[1, :]) <= self.crab_max_steer_]
        constraints += [cvxpy.abs(u[3, :]) <= self.crab_max_steer_]
        constraints += [u[1, :] == u[3, :]]
        constraints += [u[0, :] == u[2, :]]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
            oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
            oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
            ovf = np.array(u.value[0, :]).flatten()
            ovr = np.array(u.value[2, :]).flatten()
            osf = np.array(u.value[1, :]).flatten()
            osr = np.array(u.value[3, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            ox = None
            oy = None
            oyaw = None
            ovf = None
            ovr = None
            osf = None
            osr = None

        return ovf, ovr, osf, osr, ox, oy, oyaw

    def getNearestIndex(self, state, cx, cy, cyaw, pind, n_ind_search=10):

        dx = [state.x - icx for icx in cx[pind:(pind + n_ind_search)]]
        dy = [state.y - icy for icy in cy[pind:(pind + n_ind_search)]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        mind = min(d)
        ind = d.index(mind) + pind
        mind = math.sqrt(mind)
        return ind, mind

    def getReferenceTrajectory(self, state, cx, cy, cyaw, n_search_ind, pind):
        
        xref = np.zeros((self.nx_, self.horizon_ + 1))
        ind, _ = self.getNearestIndex(state, cx, cy, cyaw, pind)
        if pind >= ind:
            ind = pind
        feasible_cx = cx[ind: ind + n_search_ind]
        feasible_cy = cy[ind: ind + n_search_ind]
        feasible_cyaw = cyaw[ind: ind + n_search_ind]
        ncourse = len(feasible_cx)
        for i in range(self.horizon_ + 1):
            if i < ncourse:
                xref[0, i] = feasible_cx[i]
                xref[1, i] = feasible_cy[i]
                xref[2, i] = feasible_cyaw[i]
            else:
                xref[0, i] = feasible_cx[ncourse - 1]
                xref[1, i] = feasible_cy[ncourse - 1]
                xref[2, i] = feasible_cyaw[ncourse - 1]
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

    def retriveTrajectoryFromCSV(self, file_name):
        node_lists = []
        with open(file_name, newline='') as csvfile:
            rows = csv.reader(csvfile)
            skip_first = True
            for row in rows:
                if skip_first:
                    skip_first = False
                    continue
                node = []
                for i, element in enumerate(row):    
                    if i == 11:
                        node.append(element)
                    else:
                        node.append(float(element))
                node_lists.append(node)
        return node_lists

    def interpolateReference(self, node_lists, interpolate_num=5, mode='ackermann'):

        pts_x, pts_y, pts_yaw = [], [], []
        if mode == 'ackermann' or mode == 'diff':
            for i in range(len(node_lists) - 1):
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                if w == 0:
                    continue
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    icr = [-vy / w, vx / w]
                    yaw = from_node[2] + (to_node[2] - from_node[2]) / interpolate_num * i
                    x = (math.cos(from_node[2]) - math.cos(yaw)) * icr[0] - (math.sin(from_node[2]) - math.sin(yaw)) * icr[1] + from_node[0]
                    y = (math.sin(from_node[2]) - math.sin(yaw)) * icr[0] + (math.cos(from_node[2]) - math.cos(yaw)) * icr[1] + from_node[1]
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
            return pts_x, pts_y, pts_yaw
        
        if mode == 'crab':
            for i in range(len(node_lists) - 1):
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    yaw = from_node[2]
                    x = from_node[0] + (to_node[0] - from_node[0]) / interpolate_num * i
                    y = from_node[1] + (to_node[1] - from_node[1]) / interpolate_num * i
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
            return pts_x, pts_y, pts_yaw
        
    def interpolateReference2(self, node_lists, interpolate_num=5):

        pts_x, pts_y, pts_yaw, pts_mode = [], [], [], []
        for i in range(len(node_lists) - 1):
            mode = node_lists[i+1][11]
            if mode == 'ackermann' or mode == 'diff':
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                if w == 0:
                    continue
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    icr = [-vy / w, vx / w]
                    yaw = from_node[2] + (to_node[2] - from_node[2]) / interpolate_num * i
                    x = (math.cos(from_node[2]) - math.cos(yaw)) * icr[0] - (math.sin(from_node[2]) - math.sin(yaw)) * icr[1] + from_node[0]
                    y = (math.sin(from_node[2]) - math.sin(yaw)) * icr[0] + (math.cos(from_node[2]) - math.cos(yaw)) * icr[1] + from_node[1]
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
                    pts_mode.append(mode)
            
            elif mode == 'crab':
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    yaw = from_node[2]
                    x = from_node[0] + (to_node[0] - from_node[0]) / interpolate_num * i
                    y = from_node[1] + (to_node[1] - from_node[1]) / interpolate_num * i
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
                    pts_mode.append(mode)

        return pts_x, pts_y, pts_yaw, pts_mode

    def removeRepeatedPoints(self, cx, cy, cyaw, epsilon=0.00001):

        nx, ny, nyaw = [], [], []
        for x, y, yaw in zip(cx, cy, cyaw):
            if not nx:
                nx.append(x)
                ny.append(y)
                nyaw.append(yaw)
                continue
            dx = x - nx[-1]
            dy = y - ny[-1]
            if (dx**2 + dy**2) < epsilon:
                continue
            nx.append(x)
            ny.append(y)
            nyaw.append(yaw)
        return nx, ny, nyaw

    def splitTrajectoryWithMotionModes(self, cmode):

        trajectories_idx_group = []
        from_idx, to_idx = None, None
        current_mode = None
        for i, mode in zip(range(len(cmode) - 1), cmode):
            
            if from_idx is None:
                from_idx = i
                current_mode = mode

            if current_mode != cmode[i+1]:
                to_idx = i
                trajectories_idx_group.append([from_idx, to_idx])
                from_idx, to_idx = None, None

        return trajectories_idx_group

    def makeEightShapeTrajectory(self, size=10, n=121):
        x, y, yaw = [], [], []
        for i in range(n):
            ptx = 0.8 * math.sin(2 * math.pi / 60 * i) * size
            pty = math.sin(1 * math.pi / 60 * i) * size
            dx = 0.8 * math.cos(2 * math.pi / 60 * i) * 2 * math.pi / 60
            dy = math.cos(1 * math.pi / 60 * i) * 1 * math.pi / 60
            ptyaw = math.atan2(dy, dx)
            x.append(ptx)
            y.append(pty)
            yaw.append(ptyaw)
        return x ,y, yaw
    

def main1():

    print(__file__ + " start...")

    
    tg = TrajectoryGenerator()
    cx, cy, cyaw = tg.makeEightShapeTrajectory()
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulation(cx, cy, cyaw, initial_state)

def main2():

    print(__file__ + " start...")

    tg = TrajectoryGenerator()
    ref = tg.retriveTrajectoryFromCSV('reference.csv')
    cx, cy, cyaw = tg.interpolateReference(ref, 3)
    cx, cy, cyaw = tg.removeRepeatedPoints(cx, cy, cyaw)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulation(cx, cy, cyaw, initial_state, 'ackermann')

def main3():

    print(__file__ + " start...")

    tg = TrajectoryGenerator()
    ref = tg.retriveTrajectoryFromCSV('reference_crab.csv')
    cx, cy, cyaw = tg.interpolateReference(ref, 3, 'crab')
    cx, cy, cyaw = tg.removeRepeatedPoints(cx, cy, cyaw)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulation(cx, cy, cyaw, initial_state, 'crab')

def main4():

    print(__file__ + " start...")

    tg = TrajectoryGenerator()
    ref = tg.retriveTrajectoryFromCSV('reference_diff.csv')
    cx, cy, cyaw = tg.interpolateReference(ref, 2, 'diff')
    cx, cy, cyaw = tg.removeRepeatedPoints(cx, cy, cyaw)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulation(cx, cy, cyaw, initial_state, 'diff')

def main5():

    print(__file__ + " start...")

    tg = TrajectoryGenerator()
    ref = tg.retriveTrajectoryFromCSV('output.csv')
    cx, cy, cyaw, cmode = tg.interpolateReference2(ref, 3)
    idx_group = tg.splitTrajectoryWithMotionModes(cmode)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    mpc = MPC()
    t, x, y, yaw, vx, w = mpc.doSimulationWithAllMotionModes(idx_group, cx, cy, cyaw, cmode, initial_state)


if __name__ == '__main__':
    # main1() # 8 shaped / Ackermann Mode
    # main2() # RRT / Ackermann Mode
    # main3() # RRT / Crab Mode
    # main4() # RRT / Diff Mode
    main5()