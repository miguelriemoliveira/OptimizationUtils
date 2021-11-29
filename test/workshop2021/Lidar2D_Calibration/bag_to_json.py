#!/usr/bin/python3
import functools

import numpy as np
import rospy
from rospy_message_converter import message_converter
from sensor_msgs.msg import LaserScan
import json


def bagCallback(msg, topic, file):
    dictionary = message_converter.convert_ros_message_to_dictionary(msg)
    print('Saving message from topic ' + topic + ' to file ' + file)

    for idx, range in enumerate(dictionary['ranges']): # replace Infinity by -1
        if range == np.inf:
            dictionary['ranges'][idx] = -1

    with open(file, "w") as outfile:
        json.dump(dictionary, outfile)

    rospy.signal_shutdown('Saved topic to json')


def main():
    topic = rospy.remap_name('scan')
    file = '.' + rospy.remap_name('data')
    rospy.init_node('bag_to_json', anonymous=True)
    rospy.Subscriber(topic, LaserScan, functools.partial(bagCallback, topic=topic, file=file))

    rospy.spin()


if __name__ == '__main__':
    main()
