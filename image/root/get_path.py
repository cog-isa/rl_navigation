import sys
import os
import rospy
from nav_msgs.msg import Path

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)

def listener():
    rospy.init_node('listener', anonymous=True)   
    rospy.Subscriber('mapPath', Path, callback)
    rospy.spin()
    
if __name__ == '__main__':
    listener()
