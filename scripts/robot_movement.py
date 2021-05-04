#!/usr/bin/env python3

import rospy

import cv2, cv_bridge
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from q_learning_project.msg import RobotMoveDBToBlock

class RobotMovement(object):
    def __init__(self):
        rospy.init_node("robot_movement")

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()
        # initalize the debugging window
        # cv2.namedWindow("window", 1)
        # self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
        #                 Image, self.find_dumbbells)
        self.ready_pub = rospy.Publisher("/q_learning/ready_for_actions", Empty, queue_size=10)

        rospy.Subscriber("/q_learning/robot_action", RobotMoveDBToBlock, self.handle_robot_action)

        rospy.sleep(2)
        self.ready_pub.publish(Empty())
        print('hello2')


    def handle_robot_action(self, data):
        print(data)


    def find_dumbbells(self, msg):
        velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define the upper and lower bounds rgb
        lower_bounds = numpy.array([0, 121/2, 241/2]) 
        upper_bounds = numpy.array([60/2, 180/2, 300/2])
        rgb_lower = [[lower_bounds[i],50, 50] for i in range(3)]
        rgb_upper = [[upper_bounds[i],255, 255] for i in range(3)]

        mask = cv2.inRange(hsv, rgb_lower[self.dumbbell_target], rgb_upper[self.dumbbell_target])

        # this erases all pixels that aren't yellow
        h, w, d = image.shape
        search_top = int(3*h/4)
        search_bot = int(3*h/4 + 20)
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        # using moments() function, the center of the colored pixels is determined
        M = cv2.moments(mask)
        # if there are any colored pixels found
        if M['m00'] > 0:
                # center of the colored pixels in the image
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                # a red circle is visualized in the debugging window to indicate
                # the center point of the colored pixels
                cv2.circle(image, (cx, cy), 20, (0,0,255), -1)

                err = w/2 - cx
                k_p = 1.0 / 100.0
                self.twist.linear.x = 0.2
                self.twist.angular.z = k_p * err
                self.cmd_vel_pub.publish(self.twist)

        cv2.imshow("window", image)
        cv2.waitKey(3)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = RobotMovement()
    node.run()
