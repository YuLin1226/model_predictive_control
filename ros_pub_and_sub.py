import rospy
import tf
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from state import State
import math
import numpy as np

def angleRangeFilter(ang):

    while ang > math.pi * 2:
        ang = ang - math.pi * 2
    while ang < -math.pi * 2:
        ang = ang + math.pi * 2
    if ang < 0:
        ang = ang + math.pi * 2
    return ang

class RobotPoseSubscriber:

    def __init__(self, topic_name:str) -> None:
        
        self.node_ = State()
        self.nodes_ = []
        rospy.Subscriber(topic_name, ModelStates, self.callback)

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
        yaw = angleRangeFilter(ang=yaw)
        self.node_ = State(x=x, y=y, yaw=yaw)

    def getState(self):
        return self.node_

class CommandPublisher:

    def __init__(self, topic_name:str) -> None:

        self.publisher_ = rospy.Publisher(topic_name, Twist, queue_size = 1)

    def publishCommand(self, twist:Twist):
        
        self.publisher_.publish(twist)
        
    def prepareRobotCommand(self, vx, vy, w) -> Twist:

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = w
        return twist