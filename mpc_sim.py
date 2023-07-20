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

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True

class GBM:

    def __init__(self, wheel_base) -> None:

        self.wheel_base_ = wheel_base

        H = np.array([
            [1, 0, 0],
            [0, 1, self.wheel_base_/2],
            [1, 0, 0],
            [0, 1, -self.wheel_base_/2]
        ])
        self.kinematical_matrix_ = np.linalg.inv((H.transpose() @ H)) @ H.transpose()

    def getVelocitiesFromRobotModels(self, front_steer, front_speed, rear_steer, rear_speed):
        
        v1x = math.cos(front_steer) * front_speed
        v1y = math.sin(front_steer) * front_speed
        v2x = math.cos(rear_steer) * rear_speed
        v2y = math.sin(rear_steer) * rear_speed
        vo = np.array([
            [v1x],
            [v1y],
            [v2x],
            [v2y]
        ])
        vi = self.kinematical_matrix_ @ vo
        return vi
    
    def getRobotModelMatrice(self, theta, v_f, delta_f, v_r, delta_r):

        A = np.zeros((NX, NX))
        B = np.zeros((NX, NU))
        C = np.zeros(NX)
        # Define A
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[0, 2] = -0.5*DT*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.cos(theta) -0.5*DT*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.cos(theta)
        A[1, 2] = 0.5*DT*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.sin(theta) -0.5*DT*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.sin(theta)
        # Define B
        B[0, 0] =  1 / 2 * DT * (math.cos(delta_f) * math.cos(theta) - math.sin(delta_f) * math.sin(theta))
        B[0, 1] = -1 / 2 * DT * (math.sin(delta_f) * math.cos(theta) + math.cos(delta_f) * math.sin(theta)) * v_f
        B[0, 2] =  1 / 2 * DT * (math.cos(delta_r) * math.cos(theta) - math.sin(delta_r) * math.sin(theta))
        B[0, 3] = -1 / 2 * DT * (math.sin(delta_r) * math.cos(theta) + math.cos(delta_r) * math.sin(theta)) * v_r
        B[1, 0] =  1 / 2 * DT * (math.cos(delta_f) * math.sin(theta) + math.sin(delta_f) * math.cos(theta))
        B[1, 1] = -1 / 2 * DT * (math.sin(delta_f) * math.sin(theta) - math.cos(delta_f) * math.cos(theta)) * v_f
        B[1, 2] =  1 / 2 * DT * (math.cos(delta_r) * math.sin(theta) + math.sin(delta_r) * math.cos(theta))
        B[1, 3] = -1 / 2 * DT * (math.sin(delta_r) * math.cos(theta) - math.cos(delta_r) * math.sin(theta)) * v_r
        B[2, 0] = -1 / self.wheel_base_ * DT * math.sin(delta_r)
        B[2, 1] = -1 / self.wheel_base_ * DT * math.cos(delta_f) * v_f 
        B[2, 2] = -1 / self.wheel_base_ * DT * math.sin(delta_r)
        B[2, 3] = -1 / self.wheel_base_ * DT * math.cos(delta_r) * v_r
        # Define C
        C[0] = 1 / 2 * DT * (v_f * math.cos(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.cos(delta_r) * math.sin(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.cos(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.cos(theta) * (delta_r + theta))
        C[1] = 1 / 2 * DT * (-v_f * math.cos(delta_f) * math.cos(theta) * (delta_f + theta) - v_r * math.cos(delta_r) * math.cos(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.sin(theta) * (delta_r + theta))
        C[2] = 1 / self.wheel_base_ * DT * (-math.cos(delta_f) * delta_f * v_f + math.cos(delta_r) * delta_r * v_r)
        
        return A, B, C

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

G_GBM = GBM(wheel_base=1)


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def plot_car(x, y, yaw, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-LENGTH/2, LENGTH/2, LENGTH/2, -LENGTH/2, -LENGTH/2],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    # fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
    #                      [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    # rr_wheel = np.copy(fr_wheel)

    # fl_wheel = np.copy(fr_wheel)
    # fl_wheel[1, :] *= -1
    # rl_wheel = np.copy(rr_wheel)
    # rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    # Rot2 = np.array([[math.cos(steer), math.sin(steer)],
    #                  [-math.sin(steer), math.cos(steer)]])

    # fr_wheel = (fr_wheel.T.dot(Rot2)).T
    # fl_wheel = (fl_wheel.T.dot(Rot2)).T
    # fr_wheel[0, :] += WB
    # fl_wheel[0, :] += WB

    # fr_wheel = (fr_wheel.T.dot(Rot1)).T
    # fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    # rr_wheel = (rr_wheel.T.dot(Rot1)).T
    # rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    # fr_wheel[0, :] += x
    # fr_wheel[1, :] += y
    # rr_wheel[0, :] += x
    # rr_wheel[1, :] += y
    # fl_wheel[0, :] += x
    # fl_wheel[1, :] += y
    # rl_wheel[0, :] += x
    # rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(fr_wheel[0, :]).flatten(),
    #          np.array(fr_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(rr_wheel[0, :]).flatten(),
    #          np.array(rr_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(fl_wheel[0, :]).flatten(),
    #          np.array(fl_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(rl_wheel[0, :]).flatten(),
    #          np.array(rl_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(x, y, "*")


def update_state(state, vf, sf, vr, sr):

    V = G_GBM.getVelocitiesFromRobotModels(
            front_speed=vf, 
            front_steer=sf, 
            rear_speed=vr,
            rear_steer=sr 
        )
        
    state.x = float(state.x + V[0] * math.cos(state.yaw) * DT - V[1] * math.sin(state.yaw) * DT)
    state.y = float(state.y + V[0] * math.sin(state.yaw) * DT + V[1] * math.cos(state.yaw) * DT)
    state.yaw = float(state.yaw + V[2] * DT)
    state.vx = float(V[0])
    state.vy = float(V[1])
    state.w = float(V[2])

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


def predict_motion(x0, ov_f, ov_r, os_f, os_r, xref):
    
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[2])
    for (vf, vr, sf, sr, i) in zip(ov_f, ov_r, os_f, os_r, range(1, T + 1)):
        state = update_state(state, vf, vr, sf, sr)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, ov_f, os_f, ov_r, os_r):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw = None, None, None

    if ov_f is None or ov_r is None or os_f is None or os_r is None:
        ov_f = np.zeros(T)
        ov_r = np.zeros(T)
        os_f = np.zeros(T)
        os_r = np.zeros(T)

    uref = np.zeros((NU, T))

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ov_f, os_f, ov_r, os_r, xref)
        # pov_f, pov_r, pos_f, pos_r = ov_f[:], ov_r[:], os_f[:], os_r[:]
        
        for i in range(T):
            uref[0, i] = ov_f[i]
            uref[1, i] = os_f[i]
            uref[2, i] = ov_r[i]
            uref[3, i] = os_r[i]

        ov_f, os_f, ov_r, os_r, ox, oy, oyaw = linear_mpc_control(xref, xbar, x0, uref)
        # du = sum(abs(ov_f - pov_f)) + sum(abs(ov_r - pov_r)) + sum(abs(os_f - pos_f)) + sum(abs(os_r - pos_r))  # calc u change value
        # if du <= DU_TH:
        #     break
    else:
        print("Iterative is max iter")

    return ov_f, os_f, ov_r, os_r, ox, oy, oyaw


def linear_mpc_control(xref, xbar, x0, uref):
    
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = G_GBM.getRobotModelMatrice(theta=xbar[2, t],
                                            v_f=uref[0, t],
                                            delta_f=uref[1, t],
                                            v_r=uref[2, t],
                                            delta_r=uref[3, t])
    
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        # if t == 0:
        #     constraints += [cvxpy.abs(u[1, t] - uref[1, t]) <= MAX_DSTEER * DT]
        #     constraints += [cvxpy.abs(u[3, t] - uref[3, t]) <= MAX_DSTEER * DT]

        constraints += [u[0, t] == u[2, t]]
        constraints += [u[1, t] == -u[3, t]]
        constraints += [cvxpy.abs(u[1, t]) <= MAX_STEER]
        constraints += [cvxpy.abs(u[3, t]) <= MAX_STEER]
        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= MAX_DSTEER * DT]
            constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= MAX_DSTEER * DT]
            
    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[0, 0] == x0[0]]
    constraints += [x[1, 0] == x0[1]]
    constraints += [x[2, 0] == x0[2]]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_SPEED]
    constraints += [cvxpy.abs(u[2, :]) <= MAX_SPEED]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
        oy   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
        oyaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
        ov_f = np.array(u.value[0, :]).flatten()
        os_f = np.array(u.value[1, :]).flatten()
        ov_r  = np.array(u.value[2, :]).flatten()
        os_r  = np.array(u.value[3, :]).flatten()

    else:
        print("Error: Cannot solve mpc..")
        ox = None
        oy = None
        oyaw = None
        ov_f = None
        os_f = None
        ov_r = None
        os_r = None

    return ov_f, os_f, ov_r, os_r, ox, oy, oyaw


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]
    # dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += math.sqrt(state.vx**2 + state.vy**2) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.vx) <= STOP_SPEED and abs(state.vy) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
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

    # odelta, oa = None, None
    ov_f, os_f, ov_r, os_r = None, None, None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.yaw]  # current state

        ov_f, os_f, ov_r, os_r, ox, oy, oyaw = iterative_linear_mpc_control(
            xref, x0, dref, ov_f, os_f, ov_r, os_r)

        if os_f is not None:
            vf = ov_f[0]
            sf = os_f[0]
            vr = ov_r[0]
            sr = os_r[0]
            state = update_state(state, vf, sf, vr, sr)
            print(state.x, state.y, state.vx, state.vy)

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
            plot_car(state.x, state.y, state.yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

    return t, x, y, yaw, vx, vy, w


def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


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


def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck


def main():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_switch_back_course(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    t, x, y, yaw, vx, vy, w = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, vx, "-r", label="vx speed")
        plt.plot(t, vy, "-b", label="vx speed")
        plt.plot(t, w, "-g", label="vx speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    main()
    # main2()
