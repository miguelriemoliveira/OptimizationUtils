#!/usr/bin/python3

import rospy
from rospy_message_converter import message_converter
from sensor_msgs.msg import LaserScan
import json


def bagCallback(data):
    dictionary = message_converter.convert_ros_message_to_dictionary(data)
    print(dictionary)


def main():

    rospy.init_node('bag_to_json', anonymous=True)
    rospy.Subscriber('left_scan', LaserScan, bagCallback)

    rospy.spin()

if __name__ == '__main__':
    main()

