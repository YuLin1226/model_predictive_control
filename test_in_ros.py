#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
import tf
import time
import csv
from mpc import ModelPredictiveControl

class RobotPose:

    def __init__(self, x:float, y:float, yaw:float) -> None:

        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw

    def update(self, x:float, y:float, yaw:float) -> None:
        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw
        

class Planner:

    def __init__(self, robot_cmd_vel_topic_name:str, robot_pose_topic_name:str) -> None:

        self.mpc_ = ModelPredictiveControl()
        self.robot_pose_ = RobotPose(x=0.0, y=0.0, yaw=0.0)
        self.node_list_ = []

        self.initialization()
        rospy.init_node("mpc_planner_node", anonymous=True)
        self.time_ = self.getTimeNow()
        self.pub_ = rospy.Publisher(robot_cmd_vel_topic_name, Twist, queue_size = 1)
        self.sub_ = rospy.Subscriber(robot_pose_topic_name, ModelStates, self.updateRobotPose)

        while not rospy.is_shutdown():
            self.start()
        
        self.saveRobotTrajectory()
        rospy.spin()

    def initialization(self):

        self.mpc_.initialization(
            wheel_base=1,
            file_name="reference.csv"
        )

    def start(self):
        
        v = self.mpc_.start(current_pos_x=self.robot_pose_.x_, current_pos_y=self.robot_pose_.y_, current_pos_yaw=self.robot_pose_.yaw_)
        self.pubCommandVelocity(vx=v[0], vy=v[1], w=v[2])

    def pubCommandVelocity(self, vx, vy, w):
        
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = w
        self.pub_.publish(twist)

    def updateRobotPose(self, msg:ModelStates):

        robot_pose = msg.pose[1]
        x = robot_pose.position.x
        y = robot_pose.position.y
        
        quaternion = (
            robot_pose.orientation.x,
            robot_pose.orientation.y,
            robot_pose.orientation.z,
            robot_pose.orientation.w
            )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.robot_pose_.update(x=x, y=y, yaw=yaw)

        dt = self.getTimeNow() - self.time_
        if dt > 0.25:
            self.node_list_.append([x, y, yaw])
            self.time_ = self.getTimeNow()

    def getTimeNow(self) -> float:
        
        now = rospy.get_rostime()
        return now.to_sec()

    def saveRobotTrajectory(self):
        
        print("Save trajectory as csv.")
        with open('robot_trajectory.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'YAW'])

            for node in self.node_list_:
                writer.writerow([ 
                    node[0], 
                    node[1],
                    node[2]
                    ])
        

        print("Save prediction as csv.")
        with open('prediction.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'YAW'])

            for node in self.mpc_.predict_state_:
                writer.writerow([ 
                    node[0], 
                    node[1],
                    node[2]
                    ])




if __name__ == '__main__' :

    p = Planner(robot_cmd_vel_topic_name='/dual_wheel_steering_controller/cmd_vel', robot_pose_topic_name="/gazebo/model_states")
    # p.start()