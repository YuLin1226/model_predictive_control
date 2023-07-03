#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
import tf
import time

class RobotPoseNode:

    def __init__(self, x:float, y:float, yaw:float) -> None:

        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw
        

class RobotPoseSubscriber:

    def __init__(self, topic_name:str) -> None:
        
        self.node_list_ = []
        self.node_ = RobotPoseNode(x=0, y=0, yaw=0)

        rospy.init_node("robot_pose_subscriber_node", anonymous=True)
        rospy.Subscriber(topic_name, ModelStates, self.callback)

        while not rospy.is_shutdown():
            
            print("Add a robot pose node.")
            self.node_list_.append(self.node_)
            time.sleep(1)

        rospy.spin()

    def callback(self, msg:ModelStates):

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
        self.node_ = RobotPoseNode(x=x, y=y, yaw=yaw)

if __name__ == '__main__' :

    pass