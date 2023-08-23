#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# ================================== Library ==================================
import rospy
import time
import csv
import math
from ros_pub_and_sub import CommandPublisher, RobotPoseSubscriber
from mpc_lib import MPC
from viz import CarViz
from trajectory_gen import TrajectoryGenerator
from state import State
import numpy as np

# ================================== Class ==================================
class Writer:

    def saveTrajectoryAsCSV(self, trajectory):
        print("Save trajectory as csv.")
        with open('robot_trajectory.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'YAW'])
            for node in trajectory:
                writer.writerow([node.x, node.y, node.yaw])

# ================================== Function ==================================
def smoothYaw(yaw):

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

# ================================== Main ==================================
def main():

    rospy.init_node("mpc_planner_node", anonymous=True)
    cp = CommandPublisher(topic_name="/cmd_vel")
    rps = RobotPoseSubscriber(topic_name="/gazebo/model_states")
    viz = CarViz()
    mpc = MPC()
    tg = TrajectoryGenerator()
    csv_writer = Writer()

    # MPC Parameters
    mpc.ackermann_steer_inc_rate_ = np.deg2rad(30)
    mpc.ackermann_speed_inc_rate_ = 0.2
    mpc.ackermann_max_speed_ = 0.47
    mpc.ackermann_max_steer_ = np.deg2rad(60)
    mpc.R_  = np.diag([0.01, 0.01, 0.01, 0.01])
    mpc.Rd_ = np.diag([0.01, 0.01, 0.01, 0.01]) # Unused.
    mpc.Q_  = np.diag([1.0, 1.0, 0.1])
    mpc.Qf_ = np.diag([1.0, 1.0, 0.1])

    cx, cy, cyaw = tg.makeEightShapeTrajectory(size=10, n=481)
    cyaw = solveAtan2Continuity(cyaw)

    goal = [cx[-1], cy[-1]]
    state = rps.getState()
    target_ind, _ = mpc.getNearestIndex(state, cx, cy, cyaw, 0)
    ovf, ovr, osf, osr = None, None, None, None
    mode = 'ackermann'

    while not rospy.is_shutdown():
            
        x0 = [state.x, state.y, state.yaw] 
        xref, target_ind = mpc.getReferenceTrajectory(state, cx, cy, cyaw, 5, target_ind)
        ovf, ovr, osf, osr, ox, oy, oyaw = mpc.iterativeLMPC(xref, x0, ovf, ovr, osf, osr, mode)

        if ovf is not None:
            vfi, vri, sfi, sri = ovf[0], ovr[0], osf[0], osr[0]
            vxi, vyi, wi = mpc.gbm_.transformWheelCommandToRobotCommand(vfi, vri, sfi, sri)
            
        cmd = cp.prepareRobotCommand(vxi, vyi, wi)
        cp.publishCommand(cmd)

        if mpc.checkGoal(state, goal, target_ind, len(cx)):
            print("Goal Reached.")
            break
        
        state = rps.getState()
        rps.nodes_.append(rps.node_)

        viz.showAnimation(ox, oy, cx, cy, state.x, state.y, xref, target_ind, state)

    csv_writer.saveTrajectoryAsCSV(rps.nodes_)
    rospy.spin()

def main2():
    
    rospy.init_node("mpc_planner_node", anonymous=True)
    cp = CommandPublisher(topic_name="/dual_wheel_steering_controller/cmd_vel")
    rps = RobotPoseSubscriber(topic_name="/gazebo/model_states")
    viz = CarViz()
    mpc = MPC()
    tg = TrajectoryGenerator()
    csv_writer = Writer()

    ref = tg.retriveTrajectoryFromCSV('reference.csv')
    cx, cy, cyaw = tg.interpolateReference(ref, 3)
    cx, cy, cyaw = tg.removeRepeatedPoints(cx, cy, cyaw)

    cyaw = solveAtan2Continuity(cyaw)

    goal = [cx[-1], cy[-1]]
    state = rps.getState()
    target_ind, _ = mpc.getNearestIndex(state, cx, cy, cyaw, 0)
    ovf, ovr, osf, osr = None, None, None, None
    mode = 'ackermann'

    while not rospy.is_shutdown():
            
        x0 = [state.x, state.y, state.yaw] 
        xref, target_ind = mpc.getReferenceTrajectory(state, cx, cy, cyaw, 2, target_ind)
        ovf, ovr, osf, osr, ox, oy, oyaw = mpc.iterativeLMPC(xref, x0, ovf, ovr, osf, osr, mode)

        if ovf is not None:
            vfi, vri, sfi, sri = ovf[0], ovr[0], osf[0], osr[0]
            vxi, vyi, wi = mpc.gbm_.transformWheelCommandToRobotCommand(vfi, vri, sfi, sri)
            
        cmd = cp.prepareRobotCommand(vxi, vyi, wi)
        cp.publishCommand(cmd)

        if mpc.checkGoal(state, goal, target_ind, len(cx)):
            print("Goal Reached.")
            break
        
        state = rps.getState()
        rps.nodes_.append(rps.node_)

        viz.showAnimation(ox, oy, cx, cy, state.x, state.y, xref, target_ind, state)

    csv_writer.saveTrajectoryAsCSV(rps.nodes_)
    rospy.spin()

def main3():

    rospy.init_node("mpc_planner_node", anonymous=True)
    cp = CommandPublisher(topic_name="/dual_wheel_steering_controller/cmd_vel")
    rps = RobotPoseSubscriber(topic_name="/gazebo/model_states")
    viz = CarViz()
    mpc = MPC()
    tg = TrajectoryGenerator()
    csv_writer = Writer()

    ref = tg.retriveTrajectoryFromCSV('reference_crab.csv')
    cx, cy, cyaw = tg.interpolateReference(ref, 3, 'crab')
    cx, cy, cyaw = tg.removeRepeatedPoints(cx, cy, cyaw)
    
    cyaw = solveAtan2Continuity(cyaw)

    goal = [cx[-1], cy[-1]]
    state = rps.getState()
    target_ind, _ = mpc.getNearestIndex(state, cx, cy, cyaw, 0)
    ovf, ovr, osf, osr = None, None, None, None
    mode = 'crab'

    while not rospy.is_shutdown():
            
        x0 = [state.x, state.y, state.yaw] 
        xref, target_ind = mpc.getReferenceTrajectory(state, cx, cy, cyaw, 2, target_ind)
        ovf, ovr, osf, osr, ox, oy, oyaw = mpc.iterativeLMPC(xref, x0, ovf, ovr, osf, osr, mode)

        if ovf is not None:
            vfi, vri, sfi, sri = ovf[0], ovr[0], osf[0], osr[0]
            vxi, vyi, wi = mpc.gbm_.transformWheelCommandToRobotCommand(vfi, vri, sfi, sri)
            
        cmd = cp.prepareRobotCommand(vxi, vyi, wi)
        cp.publishCommand(cmd)

        if mpc.checkGoal(state, goal, target_ind, len(cx)):
            print("Goal Reached.")
            break
        
        state = rps.getState()
        rps.nodes_.append(rps.node_)

        viz.showAnimation(ox, oy, cx, cy, state.x, state.y, xref, target_ind, state)

    csv_writer.saveTrajectoryAsCSV(rps.nodes_)
    rospy.spin()

def main4():

    rospy.init_node("mpc_planner_node", anonymous=True)
    cp = CommandPublisher(topic_name="/dual_wheel_steering_controller/cmd_vel")
    rps = RobotPoseSubscriber(topic_name="/gazebo/model_states")
    viz = CarViz()
    mpc = MPC()
    tg = TrajectoryGenerator()
    csv_writer = Writer()

    ref = tg.retriveTrajectoryFromCSV('reference_diff.csv')
    cx, cy, cyaw = tg.interpolateReference(ref, 2, 'diff')
    cx, cy, cyaw = tg.removeRepeatedPoints(cx, cy, cyaw)
    
    cyaw = solveAtan2Continuity(cyaw)

    goal = [cx[-1], cy[-1]]
    state = rps.getState()
    target_ind, _ = mpc.getNearestIndex(state, cx, cy, cyaw, 0)
    ovf, ovr, osf, osr = None, None, None, None
    mode = 'diff'

    while not rospy.is_shutdown():
            
        x0 = [state.x, state.y, state.yaw] 
        xref, target_ind = mpc.getReferenceTrajectory(state, cx, cy, cyaw, 2, target_ind)
        ovf, ovr, osf, osr, ox, oy, oyaw = mpc.iterativeLMPC(xref, x0, ovf, ovr, osf, osr, mode)

        if ovf is not None:
            vfi, vri, sfi, sri = ovf[0], ovr[0], osf[0], osr[0]
            vxi, vyi, wi = mpc.gbm_.transformWheelCommandToRobotCommand(vfi, vri, sfi, sri)
            
        cmd = cp.prepareRobotCommand(vxi, vyi, wi)
        cp.publishCommand(cmd)

        if mpc.checkGoal(state, goal, target_ind, len(cx)):
            print("Goal Reached.")
            break
        
        state = rps.getState()
        rps.nodes_.append(rps.node_)

        viz.showAnimation(ox, oy, cx, cy, state.x, state.y, xref, target_ind, state)

    csv_writer.saveTrajectoryAsCSV(rps.nodes_)
    rospy.spin()

def main5():

    rospy.init_node("mpc_planner_node", anonymous=True)
    cp = CommandPublisher(topic_name="/dual_wheel_steering_controller/cmd_vel")
    rps = RobotPoseSubscriber(topic_name="/gazebo/model_states")
    viz = CarViz()
    mpc = MPC()
    tg = TrajectoryGenerator()
    csv_writer = Writer()

    ref = tg.retriveTrajectoryFromCSV('output.csv')
    cx, cy, cyaw, cmode = tg.interpolateReference2(ref, 3)
    idx_group = tg.splitTrajectoryWithMotionModes(cmode)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])

    mpc = MPC()
    t, x, y, yaw, vx, w = mpc.doSimulationWithAllMotionModes(idx_group, cx, cy, cyaw, cmode, initial_state)

# ================================== Run Code ==================================
if __name__ == '__main__' :
    main()
    
    