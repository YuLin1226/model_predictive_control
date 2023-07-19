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
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import cubic_spline_planner

NX = 3  
NU = 2
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 0.01])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 1  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]

MAX_V_SPEED = 0.47  # maximum speed [m/s]
MAX_W_SPEED = 3.77  # minimum speed [m/s]
DIFF_V_SPEED = 0.2
DIFF_W_SPEED = 1.0


show_animation = True

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.0, w=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
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


def update_state(state, v, w):

    state.x = state.x + v * math.cos(state.yaw) * DT
    state.y = state.y + v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + w * DT
    state.vx = v
    state.w = w

    return state


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


def predict_motion(x0, v, w, xref):
    
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[2])
    for (vi, wi, i) in zip(v, w, range(1, T + 1)):
        state = update_state(state, vi, wi)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, ov, ow):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw = None, None, None

    if ov is None or ow is None:
        ov = [0.3] * T
        ow = [0.0] * T

    uref = np.zeros((NU, T))

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ov, ow, xref)
        print("xbar", xbar)
        print("xref", xref)
        for i in range(T):
            uref[0, i] = ov[i]
            uref[1, i] = ow[i]

        ov, ow, ox, oy, oyaw = linear_mpc_control(xref, xbar, x0, uref)
        
    else:
        print("Iterative is max iter")

    return ov, ow, ox, oy, oyaw


def linear_mpc_control(xref, xbar, x0, uref):
    
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(vr=uref[0, t], theta_r=xbar[2, t])
    
        # constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
        if t < (T - 1):
            constraints += [cvxpy.abs(u[0, t+1] - u[0, t]) <= DIFF_V_SPEED]
            constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= DIFF_W_SPEED]


    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_V_SPEED]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_W_SPEED]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
        oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
        oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
        ov = np.array(u.value[0, :]).flatten()
        ow = np.array(u.value[1, :]).flatten()

    else:
        print("Error: Cannot solve mpc..")
        ox = None
        oy = None
        oyaw = None
        ov = None
        ow = None

    return ov, ow, ox, oy, oyaw


def get_linear_model_matrix(vr, theta_r):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[0, 2] = -vr * math.sin(theta_r) * DT
    A[1, 2] =  vr * math.cos(theta_r) * DT
    
    B = np.zeros((NX, NU))
    B[0, 0] = math.cos(theta_r) * DT
    B[1, 0] = math.sin(theta_r) * DT
    B[2, 1] = DT

    C = np.zeros(NX)
    C[0] =  vr * math.sin(theta_r) * theta_r * DT
    C[1] = -vr * math.cos(theta_r) * theta_r * DT

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
    w = [state.w]

    t = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    ov, ow = None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        # xref, target_ind = calc_ref_trajectory(
        #     state, cx, cy, cyaw, dl, target_ind)

        xref, target_ind = calc_ref_trajectory2(
            state, cx, cy, cyaw, 2, target_ind)

        x0 = [state.x, state.y, state.yaw]  # current state

        ov, ow, ox, oy, oyaw = iterative_linear_mpc_control(
            xref, x0, ov, ow)

        if ov is not None:
            vi, wi = ov[0], ow[0]
            state = update_state(state, vi, wi)
            
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        vx.append(state.vx)
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

    # if show_animation:  # pragma: no cover
    #     plt.close("all")
    #     plt.subplots()
    #     plt.plot(cx, cy, "-r", label="spline")
    #     plt.plot(x, y, "-g", label="tracking")
    #     plt.grid(True)
    #     plt.axis("equal")
    #     plt.xlabel("x[m]")
    #     plt.ylabel("y[m]")
    #     plt.legend()

    #     plt.subplots()
    #     plt.plot(t, vx, "-r", label="vx speed")
    #     plt.plot(t, w, "-g", label="w speed")
    #     plt.grid(True)
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Speed [kmh]")

    #     plt.show()

def main2():

    x, y, yaw = makeEightShapeTrajectory(400, 10)
    plt.plot(x, y, 'r.')
    plt.show()


if __name__ == '__main__':
    main()
    # main2()
    
