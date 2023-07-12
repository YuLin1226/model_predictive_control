#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import math
import csv

class PoseNode:

    def __init__(self, x, y, yaw) -> None:
        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw

class RobotPoseAnalyst:

    def __init__(self, robot_pose_file_name: str, predicted_pose_file_name:str, planner_pose_file_name:str) -> None:

        self.robot_pose_file_name_ = robot_pose_file_name
        self.predicted_pose_file_name_ = predicted_pose_file_name
        self.planner_pose_file_name_ = planner_pose_file_name
        self.robot_pose_lists_ = []
        self.predicted_pose_lists_ = []
        self.planner_pose_lists_ = []
        self.fig, self.ax = plt.subplots()
        
    def doAnalysis(self):

        self.getPredictedPose()
        self.getRobotPose()
        self.getPlannerPose()
        self.plotPoses()
        self.vizOn()

    def getRobotPose(self):

        with open(self.robot_pose_file_name_, newline='') as csvfile:
            rows = csv.reader(csvfile)
            ith = 0
            for row in rows:
                if ith == 0:
                    ith += 1
                    continue
                x = float(row[0])
                y = float(row[1])
                yaw = float(row[2])
                robot_pose_node = PoseNode(x=x, y=y, yaw=yaw)
                self.robot_pose_lists_.append(robot_pose_node)

    def getPredictedPose(self):

        with open(self.predicted_pose_file_name_, newline='') as csvfile:
            rows = csv.reader(csvfile)
            ith = 0
            for row in rows:
                if ith == 0:
                    ith += 1
                    continue
                x = float(row[0])
                y = float(row[1])
                yaw = float(row[2])
                predicted_pose_node = PoseNode(x=x, y=y, yaw=yaw)
                self.predicted_pose_lists_.append(predicted_pose_node)

    def getPlannerPose(self):

        with open(self.planner_pose_file_name_, newline='') as csvfile:
            rows = csv.reader(csvfile)
            ith = 0
            for row in rows:
                if ith == 0:
                    ith += 1
                    continue
                x = float(row[0])
                y = float(row[1])
                yaw = float(row[2])
                planner_pose_node = PoseNode(x=x, y=y, yaw=yaw)
                self.planner_pose_lists_.append(planner_pose_node)

    def plotPoses(self):
        x_list = []
        y_list = []
        for n in self.robot_pose_lists_:
            x = n.x_
            y = n.y_
            arrow_x = 0.5 * math.cos(n.yaw_)
            arrow_y = 0.5 * math.sin(n.yaw_)
            # self.ax.arrow(n.x_, n.y_, arrow_x, arrow_y, width=0.1, head_width=0.25, color="red")
            x_list.append(x)
            y_list.append(y)
        self.ax.plot(x_list, y_list, 'r.')

        x_list = []
        y_list = []
        for n in self.predicted_pose_lists_:
            x = n.x_
            y = n.y_            
            arrow_x = 0.5 * math.cos(n.yaw_)
            arrow_y = 0.5 * math.sin(n.yaw_)
            # self.ax.arrow(n.x_, n.y_, arrow_x, arrow_y, width=0.1, head_width=0.25, color="blue")
            x_list.append(x)
            y_list.append(y)
        self.ax.plot(x_list, y_list, 'b-')

        x_list = []
        y_list = []
        for n in self.planner_pose_lists_:
            x = n.x_
            y = n.y_            
            arrow_x = 0.5 * math.cos(n.yaw_)
            arrow_y = 0.5 * math.sin(n.yaw_)
            self.ax.arrow(n.x_, n.y_, arrow_x, arrow_y, width=0.1, head_width=0.25, color="green")
            x_list.append(x)
            y_list.append(y)
        self.ax.plot(x_list, y_list, 'g-')

    def vizOn(self):
        plt.show()

    def vizOff(self):
        plt.close()

if __name__ == '__main__' :

    RPA = RobotPoseAnalyst("robot_trajectory.csv", "prediction.csv", "reference.csv")
    RPA.doAnalysis()