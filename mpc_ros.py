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

# ================================== Main ==================================
def main():

    rospy.init_node("mpc_planner_node", anonymous=True)
    cp = CommandPublisher(topic_name="/dual_wheel_steering_controller/cmd_vel")
    rps = RobotPoseSubscriber(topic_name="/gazebo/model_states")
    viz = CarViz()
    mpc = MPC()
    tg = TrajectoryGenerator()
    csv_writer = Writer()

    cx, cy, cyaw = tg.makeEightShapeTrajectory(size=10, n=121)
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


# ================================== Run Code ==================================
if __name__ == '__main__' :
    main()
    
    