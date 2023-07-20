#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import pathlib
import csv
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import cubic_spline_planner

NX = 3  
NU = 4
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01, 0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 0.01, 0.01, 0.01])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 1  # Max iteration
DU_TH = 0.1  # iteration finish param

N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]

MAX_V_SPEED = 0.47  # maximum speed [m/s]
DIFF_V_SPEED = 0.2
MAX_STEER = np.deg2rad(60)
DIFF_STEER = np.deg2rad(30)



show_animation = True

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, w=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.w = w
        self.predelta = None

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def plot_car(x, y, yaw, cabcolor="-r", truckcolor="-k"): 
    
    outline = np.array([[-LENGTH/2, LENGTH/2, LENGTH/2, -LENGTH/2, -LENGTH/2],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])

    outline = (outline.T.dot(Rot1)).T
    outline[0, :] += x
    outline[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)




def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def update_state(state, vx, vy, w):

    state.x = state.x + vx * math.cos(state.yaw) * DT - vy * math.sin(state.yaw) * DT
    state.y = state.y + vx * math.sin(state.yaw) * DT + vy * math.cos(state.yaw) * DT
    state.yaw = state.yaw + w * DT
    state.vx = vx
    state.vy = vy
    state.w = w

    return state

def predict_motion(x0, vf, vr, sf, sr, xref):
    
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[2])
    for (vfi, vri, sfi, sri, i) in zip(vf, vr, sf, sr, range(1, T + 1)):
        vx, vy, w = transfrom_gbm_command(vfi, vri, sfi, sri)
        state = update_state(state, vx, vy, w)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.yaw

    return xbar


def transfrom_gbm_command(vf, vr, sf, sr, wheel_base=1):

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


def iterative_linear_mpc_control(xref, x0, ovf, ovr, osf, osr):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw = None, None, None

    if ovf is None or ovr is None or osf is None or osr is None:
        ovf = [0.0] * T
        ovr = [0.0] * T
        osf = [0.0] * T
        osr = [0.0] * T

    uref = np.zeros((NU, T))

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ovf, ovr, osf, osr, xref)

        for i in range(T):
            uref[0, i] = ovf[i]
            uref[2, i] = ovr[i]
            uref[1, i] = osf[i]
            uref[3, i] = osr[i]
            
        # ovf, ovr, osf, osr, ox, oy, oyaw = linear_mpc_control_ackermann(xref, xbar, x0, uref)
        # ovf, ovr, osf, osr, ox, oy, oyaw = linear_mpc_control_diff(xref, xbar, x0, uref)
        ovf, ovr, osf, osr, ox, oy, oyaw = linear_mpc_control_crab(xref, xbar, x0, uref)
        
    else:
        print("Iterative is max iter")

    return ovf, ovr, osf, osr, ox, oy, oyaw


def linear_mpc_control_ackermann(xref, xbar, x0, uref):
    
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            theta=xbar[2, t],
            v_f=uref[0, t],
            v_r=uref[2, t],
            delta_f=uref[1, t],
            delta_r=uref[3, t])
    
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= DIFF_V_SPEED * DT]
            constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= DIFF_V_SPEED * DT]
            constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= DIFF_STEER * DT]
            constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= DIFF_STEER * DT]

        if t == 0:
            constraints += [cvxpy.abs(uref[1, t] - u[1, t]) <= DIFF_STEER * DT]
            constraints += [cvxpy.abs(uref[3, t] - u[3, t]) <= DIFF_STEER * DT]


    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_V_SPEED]
    constraints += [cvxpy.abs(u[2, :]) <= MAX_V_SPEED]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]
    constraints += [cvxpy.abs(u[3, :]) <= MAX_STEER]
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

def linear_mpc_control_crab(xref, xbar, x0, uref):
    
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            theta=xbar[2, t],
            v_f=uref[0, t],
            v_r=uref[2, t],
            delta_f=uref[1, t],
            delta_r=uref[3, t])
    
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= DIFF_V_SPEED * DT]
            constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= DIFF_V_SPEED * DT]
            constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= DIFF_STEER * DT]
            constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= DIFF_STEER * DT]

        if t == 0:
            constraints += [cvxpy.abs(uref[1, t] - u[1, t]) <= DIFF_STEER * DT]
            constraints += [cvxpy.abs(uref[3, t] - u[3, t]) <= DIFF_STEER * DT]


    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_V_SPEED]
    constraints += [cvxpy.abs(u[2, :]) <= MAX_V_SPEED]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]
    constraints += [cvxpy.abs(u[3, :]) <= MAX_STEER]
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

def linear_mpc_control_diff(xref, xbar, x0, uref):
    
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            theta=xbar[2, t],
            v_f=uref[0, t],
            v_r=uref[2, t],
            delta_f=uref[1, t],
            delta_r=uref[3, t])
    
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= DIFF_V_SPEED * DT]
            constraints += [cvxpy.abs(u[2, t+1] - u[2, t]) <= DIFF_V_SPEED * DT]
            constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= DIFF_STEER * DT]
            constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= DIFF_STEER * DT]

        if t == 0:
            constraints += [cvxpy.abs(uref[1, t] - u[1, t]) <= DIFF_STEER * DT]
            constraints += [cvxpy.abs(uref[3, t] - u[3, t]) <= DIFF_STEER * DT]


    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_V_SPEED]
    constraints += [cvxpy.abs(u[2, :]) <= MAX_V_SPEED]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]
    constraints += [cvxpy.abs(u[3, :]) <= MAX_STEER]
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

def get_linear_model_matrix(theta, v_f, delta_f, v_r, delta_r, wheel_base=1):

    # Define A
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[0, 2] = -0.5*DT*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.sin(theta) -0.5*DT*(v_f*math.sin(delta_f)+v_r*math.sin(delta_r))*math.cos(theta)
    A[1, 2] = 0.5*DT*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.cos(theta) -0.5*DT*(v_f*math.sin(delta_f)+v_r*math.sin(delta_r))*math.sin(theta)
    # Define B
    B = np.zeros((NX, NU))
    B[0, 0] =  1 / 2 * DT * (math.cos(delta_f) * math.cos(theta) - math.sin(delta_f) * math.sin(theta))
    B[0, 1] = -1 / 2 * DT * (math.sin(delta_f) * math.cos(theta) + math.cos(delta_f) * math.sin(theta)) * v_f
    B[0, 2] =  1 / 2 * DT * (math.cos(delta_r) * math.cos(theta) - math.sin(delta_r) * math.sin(theta))
    B[0, 3] = -1 / 2 * DT * (math.sin(delta_r) * math.cos(theta) + math.cos(delta_r) * math.sin(theta)) * v_r
    B[1, 0] =  1 / 2 * DT * (math.cos(delta_f) * math.sin(theta) + math.sin(delta_f) * math.cos(theta))
    B[1, 1] = -1 / 2 * DT * (math.sin(delta_f) * math.sin(theta) - math.cos(delta_f) * math.cos(theta)) * v_f
    B[1, 2] =  1 / 2 * DT * (math.cos(delta_r) * math.sin(theta) + math.sin(delta_r) * math.cos(theta))
    B[1, 3] = -1 / 2 * DT * (math.sin(delta_r) * math.cos(theta) - math.cos(delta_r) * math.sin(theta)) * v_r
    B[2, 0] =  1 / wheel_base * DT * math.sin(delta_r)
    B[2, 1] =  1 / wheel_base * DT * math.cos(delta_f) * v_f 
    B[2, 2] = -1 / wheel_base * DT * math.sin(delta_r)
    B[2, 3] = -1 / wheel_base * DT * math.cos(delta_r) * v_r
    # Define C
    C = np.zeros(NX)
    C[0] = 1 / 2 * DT * (v_f * math.cos(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.cos(delta_r) * math.sin(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.cos(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.cos(theta) * (delta_r + theta))
    C[1] = 1 / 2 * DT * (-v_f * math.cos(delta_f) * math.cos(theta) * (delta_f + theta) - v_r * math.cos(delta_r) * math.cos(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.sin(theta) * (delta_r + theta))
    C[2] = 1 / wheel_base * DT * (-math.cos(delta_f) * delta_f * v_f + math.cos(delta_r) * delta_r * v_r)
    
    return A, B, C


def calc_ref_trajectory(state, cx, cy, cyaw, dl, pind):
    xref = np.zeros((NX, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.vx) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]

    return xref, ind

def calc_ref_trajectory2(state, cx, cy, cyaw, n_search_ind, pind):
    xref = np.zeros((NX, T + 1))

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    feasible_cx = cx[ind: ind + n_search_ind]
    feasible_cy = cy[ind: ind + n_search_ind]
    feasible_cyaw = cyaw[ind: ind + n_search_ind]

    ncourse = len(feasible_cx)

    for i in range(T + 1):

        if i < ncourse:
            xref[0, i] = feasible_cx[i]
            xref[1, i] = feasible_cy[i]
            xref[2, i] = feasible_cyaw[i]
        else:
            xref[0, i] = feasible_cx[ncourse - 1]
            xref[1, i] = feasible_cy[ncourse - 1]
            xref[2, i] = feasible_cyaw[ncourse - 1]

    return xref, ind


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.vx) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def do_simulation(cx, cy, cyaw, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

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
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    ovf, ovr, osf, osr = None, None, None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        # xref, target_ind = calc_ref_trajectory(
        #     state, cx, cy, cyaw, dl, target_ind)

        xref, target_ind = calc_ref_trajectory2(
            state, cx, cy, cyaw, 2, target_ind)

        x0 = [state.x, state.y, state.yaw]  # current state

        ovf, ovr, osf, osr, ox, oy, oyaw = iterative_linear_mpc_control(
            xref, x0, ovf, ovr, osf, osr)

        if ovf is not None:
            vfi, vri, sfi, sri = ovf[0], ovr[0], osf[0], osr[0]
            vxi, vyi, wi = transfrom_gbm_command(vfi, vri, sfi, sri)
            state = update_state(state, vxi, vyi, wi)
            
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        vx.append(state.vx)
        vy.append(state.vy)
        w.append(state.w)
        t.append(time)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            # plot_car(state.x, state.y, state.yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

    return t, x, y, yaw, vx, w


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


def makeEightShapeTrajectory(n, size):

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

def main():
    print(__file__ + " start!!")

    dl = 0.5  # course tick
    cx, cy, cyaw = makeEightShapeTrajectory(400, 10)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    t, x, y, yaw, vx, w = do_simulation(
        cx, cy, cyaw, dl, initial_state)

def main2():

    print(__file__ + " start!!")

    dl = 0.5

    reference = retriveReferenceFromCSV('reference.csv')
    cx, cy, cyaw = interpolateReference(reference, 3)
    cx, cy, cyaw = removeRepeatedPoints(cx, cy, cyaw)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    t, x, y, yaw, vx, w = do_simulation(
        cx, cy, cyaw, dl, initial_state)


def main3():

    print(__file__ + " start!!")

    dl = 0.5

    reference = retriveReferenceFromCSV('reference_crab.csv')

    cx, cy, cyaw = interpolateReference(reference, 3, 'crab')
    cx, cy, cyaw = removeRepeatedPoints(cx, cy, cyaw)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    cx.pop(0)
    cy.pop(0)
    cyaw.pop(0)

    t, x, y, yaw, vx, w = do_simulation(
        cx, cy, cyaw, dl, initial_state)

def retriveReferenceFromCSV(file_name):
    
    node_lists = []

    with open(file_name, newline='') as csvfile:
        rows = csv.reader(csvfile)
        n = 0
        for row in rows:
            if n == 0:
                n += 1
                continue

            node_lists.append([float(e) for e in row])
        # Node List Info:
        # x | y | yaw | vx | vy | w | front_wheel: angle | front_wheel: speed | rear_wheel: angle | rear_wheel: speed  | time_stamp

    return node_lists

def interpolateReference(node_lists, interpolate_num=5, mode='ackermann'):

    pts_x, pts_y, pts_yaw = [], [], []

    if mode == 'ackermann':

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

def removeRepeatedPoints(cx, cy, cyaw, epsilon=0.00001):

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


if __name__ == '__main__':
    # main() # 8 shaped / Ackermann Mode
    # main2() # RRT / Ackermann Mode
    main3() # RRT / Crab Mode
    
