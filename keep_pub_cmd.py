#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import rospy
import tf
from geometry_msgs.msg import Twist

class KeepPublishCommand:

    def __init__(self, pub_topic_name:str, sub_topic_name:str) -> None:
        
        self.last_twist_cmd_ = Twist()
        self.last_received_time_ = self.getTimeNow()
        self.publisher_ = rospy.Publisher(pub_topic_name, Twist, queue_size = 1)
        rospy.Subscriber(sub_topic_name, Twist, self.callback)

    def getTimeNow(self):
        now = rospy.get_rostime()
        return now.to_sec()

    def callback(self, msg:Twist):
        self.last_twist_cmd_ = msg
        self.last_received_time_ = self.getTimeNow()

    def publishCommand(self):
        time_now = self.getTimeNow()
        if time_now - self.last_received_time_ < 0.5:
            self.publisher_.publish(self.last_twist_cmd_)

if __name__ == '__main__' :
    
    rospy.init_node("mpc_planner_node", anonymous=True)
    KPC = KeepPublishCommand(
        pub_topic_name="/dual_wheel_steering_controller/cmd_vel",
        sub_topic_name="/cmd_vel")
    while not rospy.is_shutdown():
        KPC.publishCommand()
    rospy.spin()