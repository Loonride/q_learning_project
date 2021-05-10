#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

from std_msgs.msg import Empty
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from q_learning_project.msg import RobotMoveDBToBlock
import keras_ocr

class RobotMovement(object):
    def __init__(self):
        rospy.init_node("robot_movement")
        self.initalized = False

        # download pre-trained model
        self.keras_pipeline = keras_ocr.pipeline.Pipeline()

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()
        # initalize the debugging window
        # cv2.namedWindow("window", 1)
        # set self.front_distance to distance to nearest object in front of robot
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.getDistance)
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.complete_action)
        self.ready_pub = rospy.Publisher("/q_learning/ready_for_actions", Empty, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.action_queue = []
        self.front_distance = 100
        self.carrying_db = False
        rospy.Subscriber("/q_learning/robot_action", RobotMoveDBToBlock, self.handle_robot_action)
        rospy.sleep(1)
        self.ready_pub.publish(Empty())

        self.initalized = True


    def handle_robot_action(self, data):
        self.action_queue.append(data)

    def getDistance(self, msg):
        self.front_distance = msg.ranges[0]

    def complete_action(self, msg):
        if (not self.initalized):
            return

        if len(self.action_queue) == 0:
            return

        if self.carrying_db:
            self.find_number(msg)
            return

        target = self.action_queue[0]
        color = target.robot_db # "red", "blue", "green"
        self.block_id = target.block_id # 1, 2, 3

        self.dumbbell_target = np.where(np.asarray(["red","green","blue"])==color)[0][0]

        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define the upper and lower bounds rgb
        lower_bounds = np.array([0, 121/2, 241/2]) 
        upper_bounds = np.array([300000, 180/2, 300/2])
        rgb_lower = [np.asarray([lower_bounds[i],20, 20]) for i in range(3)]
        rgb_upper = [np.asarray([upper_bounds[i],255, 255]) for i in range(3)]
        
        print("target:", rgb_lower[self.dumbbell_target])
        mask = cv2.inRange(hsv, rgb_lower[self.dumbbell_target], rgb_upper[self.dumbbell_target])
        print("Mask:",mask)
        # this erases all pixels that aren't desired color
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
                if err > .05: 
                    self.set_v(.2, k_p*err)
                elif self.front_distance > .8: 
                    self.set_v(.2, k_p*err)
                else: 
                    self.set_v(0,0)
                    #pick up dumbbell
                    self.carrying_db = True
        else: 
            self.set_v(0,.2)
        cv2.imshow("window", image)
        cv2.waitKey(3)

    def find_number(self, msg):
        """ Find block with target ID
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image = np.asarray(image)
        prediction_groups = self.keras_pipeline.recognize([image])
        print("Predicted groups:", prediction_groups)
        if self.block_id in prediction_groups:
            set_v(1,0)
        else: 
            set_v(0,.1)

    def set_v(self, velocity, angular_velocity):
        """ The current velocity and angular velocity of the robot are set here
        """
        v1 = Vector3(velocity, 0.0, 0.0)
        v2 = Vector3(0.0, 0.0, angular_velocity)
        t = Twist(v1, v2)
        self.cmd_vel_pub.publish(t)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = RobotMovement()
    node.run()
