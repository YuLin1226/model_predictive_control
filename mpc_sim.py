#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import cvxpy
import math
import numpy as np
import sys
from library.polynomial import Polynomial
from library.visualization import CarViz
from library.general_bicycle_model import GeneralBicycleModel
from library.trajectory_generator import TrajectoryGenerator

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

                if mode == 'ackermann' or mode == 'ackermann/crab':
                    ovf, ovr, osf, osr, ox, oy, oyaw = self.doLMPC_Ackermann(xref, xbar, x0, uref)
                elif mode == 'differential' or mode == 'differential/crab':
                    ovf, ovr, osf, osr, ox, oy, oyaw = self.doLMPC_Differential(xref, xbar, x0, uref)
                elif mode == 'crab' or mode == 'crab/differential':
                    ovf, ovr, osf, osr, ox, oy, oyaw = self.doLMPC_Crab(xref, xbar, x0, uref)
                else:
                    print("Mode " + mode + " not defined. ")
            
        return ovf, ovr, osf, osr, ox, oy, oyaw

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

def main2():

    print(__file__ + " start...")

    tg = TrajectoryGenerator()
    node_lists = tg.retriveTrajectoryFromCSV('g2_ackermann.csv')
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.interpolateTrajectory(node_lists)
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.makeEquivalentDistanceBetweenPoints(cx, cy, cyaw, cmode, cvf, csf, cvr, csr, 0.05)
    # cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.compressTrajectory(cx, cy, cyaw, cmode, cvf, csf, cvr, csr, 15)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulation(cx, cy, cyaw, initial_state, 'ackermann')

def main3():

    print(__file__ + " start...")

    tg = TrajectoryGenerator()
    node_lists = tg.retriveTrajectoryFromCSV('g2_crab.csv')
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.interpolateTrajectory(node_lists)
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.compressTrajectory(cx, cy, cyaw, cmode, cvf, csf, cvr, csr)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulation(cx, cy, cyaw, initial_state, 'crab')

def main4():

    print(__file__ + " start...")

    tg = TrajectoryGenerator()
    node_lists = tg.retriveTrajectoryFromCSV('g2_diff.csv')
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.interpolateTrajectory(node_lists)
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.compressTrajectory(cx, cy, cyaw, cmode, cvf, csf, cvr, csr)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    mpc = MPC()
    t, x, y, yaw, vx, vy, w, state = mpc.doSimulation(cx, cy, cyaw, initial_state, 'differential')

def main5():
    print(__file__ + " start...")
    tg = TrajectoryGenerator()
    node_lists = tg.retriveTrajectoryFromCSV('g2_cmd_path.csv')
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.interpolateTrajectory(node_lists)
    cx, cy, cyaw, cmode, cvf, csf, cvr, csr = tg.compressTrajectory(cx, cy, cyaw, cmode, cvf, csf, cvr, csr)
    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)
    cmode.pop(0)

    idx_group = tg.splitTrajectoryWithMotionModes(cmode)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    mpc = MPC()

    t, x, y, yaw, vx, w = mpc.doSimulationWithAllMotionModes(idx_group, cx, cy, cyaw, cmode, initial_state)

if __name__ == '__main__':
    
    # main2() # RRT / Ackermann Mode
    # main3() # RRT / Crab Mode
    main4() # RRT / Diff Mode
    # main5() # RRT / All Modes