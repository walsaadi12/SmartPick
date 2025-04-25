#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

class IntelligentGrasp:
    def __init__(self):
        rospy.init_node('intelligent_grasp_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.target_pub = rospy.Publisher('/grasp_target', Point, queue_size=10)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red detection
        red_lower = (0, 120, 70)
        red_upper = (10, 255, 255)
        mask_red = cv2.inRange(hsv, red_lower, red_upper)

        # Blue detection
        blue_lower = (100, 150, 0)
        blue_upper = (140, 255, 255)
        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

        # Find contours
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours_red:
            c = max(contours_red, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.target_pub.publish(Point(x=cx, y=cy, z=0))
                rospy.loginfo("Red cube detected at (%d, %d)", cx, cy)

        if contours_blue:
            c = max(contours_blue, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.target_pub.publish(Point(x=cx, y=cy, z=1))
                rospy.loginfo("Blue cube detected at (%d, %d)", cx, cy)

if __name__ == '__main__':
    try:
        IntelligentGrasp()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
