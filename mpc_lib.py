#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import cvxpy
import math
import numpy as np
from state import State


    
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

class MPC:

    def __init__(self, horizon=5, nx=3, nu=4, xy_tolerance=1.5, stop_speed=0.2, show_animation=True) -> None:
        
        self.horizon_ = horizon
        self.nx_ = nx
        self.nu_ = nu
        self.gbm_ = GeneralBicycleModel(wheel_base=1, nx=nx, nu=nu)
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

    def predictMotion(self, x0, vf, vr, sf, sr, xref):
        
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[2])
        for (vfi, vri, sfi, sri, i) in zip(vf, vr, sf, sr, range(1, self.horizon_ + 1)):
            vx, vy, w = self.gbm_.transformWheelCommandToRobotCommand(vfi, vri, sfi, sri)
            state = self.gbm_.updateState(state, vx, vy, w)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.yaw

        return xbar

    def iterativeLMPC(self, xref, x0, ovf, ovr, osf, osr, mode='ackermann', max_iter=1):
        
        ox, oy, oyaw = None, None, None
        if ovf is None or ovr is None or osf is None or osr is None:
            ovf = [0.0] * self.horizon_
            ovr = [0.0] * self.horizon_
            osf = [0.0] * self.horizon_
            osr = [0.0] * self.horizon_
        uref = np.zeros((self.nu_, self.horizon_))
        for i in range(max_iter):
            xbar = self.predictMotion(x0, ovf, ovr, osf, osr, xref)
            for i in range(self.horizon_):
                uref[0, i] = ovf[i]
                uref[2, i] = ovr[i]
                uref[1, i] = osf[i]
                uref[3, i] = osr[i]

                if mode == 'ackermann':
                    ovf, ovr, osf, osr, ox, oy, oyaw = self.doLMPC_Ackermann(xref, xbar, x0, uref)
                elif mode == 'diff':
                    ovf, ovr, osf, osr, ox, oy, oyaw = self.doLMPC_Differential(xref, xbar, x0, uref)
                elif mode == 'crab':
                    ovf, ovr, osf, osr, ox, oy, oyaw = self.doLMPC_Crab(xref, xbar, x0, uref)
                else:
                    print("Mode not defined. ")
            
        return ovf, ovr, osf, osr, ox, oy, oyaw

    def doLMPC_Ackermann(self, xref, xbar, x0, uref, dt=0.1):
        
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

    def doLMPC_Differential(self, xref, xbar, x0, uref, dt=0.1):
        
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

    def doLMPC_Crab(self, xref, xbar, x0, uref, dt=0.1):

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
