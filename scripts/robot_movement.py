#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import moveit_commander
import math
from std_msgs.msg import Empty
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from q_learning_project.msg import RobotMoveDBToBlock
import keras_ocr

class RobotMovement(object):
    def __init__(self):
        rospy.init_node("robot_movement")
        self.initialized = False

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
        self.carrying_db_successful = False
        rospy.Subscriber("/q_learning/robot_action", RobotMoveDBToBlock, self.handle_robot_action)
        rospy.sleep(1)
        self.ready_pub.publish(Empty())

        self.turning = 0
        self.arm_initialized = False

        print("Starting arm")
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        print("Initialized")        
        self.initialized = True


    def handle_robot_action(self, data):
        self.action_queue.append(data)

    def getDistance(self, msg):
        self.front_distance = msg.ranges[0]

    def complete_action(self, msg):
        if (not self.initialized):
            return

        if len(self.action_queue) == 0:
            return

        if self.carrying_db:
            print("HERE") #Need to go to office hours to ask about this
            # if self.turning==0:
            #     print("TURNING") 
            #     self.set_v(0,3.1415/2)
            #     rospy.sleep(2.)
            #     self.turning=1
            # else:
            #     self.set_v(0,0)
            #     self.find_number(msg)
            #     image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            #     #cv2.imshow("window", image)
            #     #cv2.waitKey(0)
            self.initialized = False
            return

        if not self.arm_initialized:
            self.initialize_arm()
            self.arm_initialized = True

        target = self.action_queue[0]
        color = target.robot_db # "red", "blue", "green"
        self.block_id = target.block_id # 1, 2, 3

        self.dumbbell_target = np.where(np.asarray(["red","green","blue"])==color)[0][0]

        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define the upper and lower bounds rgb
        lower_bounds = np.array([0, 121/2, 241/2]) 
        upper_bounds = np.array([20, 180/2, 300/2])
        rgb_lower = [np.asarray([lower_bounds[i],20, 20]) for i in range(3)]
        rgb_upper = [np.asarray([upper_bounds[i],255, 255]) for i in range(3)]
        
        mask = cv2.inRange(hsv, rgb_lower[self.dumbbell_target], rgb_upper[self.dumbbell_target])
        if color=="red": 
            mask2 = cv2.inRange(hsv, np.asarray([170/2, 20, 20]), np.asarray([180/2, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
        #cv2.imshow("window", mask)
        #cv2.waitKey(0)
        # this erases all pixels that aren't desired color
        h, w, d = image.shape

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
                k_p = .003
                if self.front_distance > .22:
                    if err > .05: 
                        self.set_v(.05, k_p*err)
                    else:
                        self.set_v(.2, k_p*err)
                else:
                    self.set_v(0,0)
                    #pick up dumbbell
                    self.pickup_db()
                    self.carrying_db = True
        #else: 
        #    self.set_v(0,.2)
        #cv2.imshow("window", mask)
        cv2.waitKey(3)


    def initialize_arm(self):
        print('Initializing Arm')
        gripper_joint_open = [0.01, 0.01]
        self.move_group_gripper.go(gripper_joint_open, wait=True)
        self.move_group_gripper.stop()

        # wait=True ensures that the movement is synchronous
        self.move_group_arm.go([0.0,
                    math.radians(50.0),
                    math.radians(-30.0),
                    math.radians(-20.0)], wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group_arm.stop()
        print('Arm Initialized')

    def pickup_db(self):
        """ Pickup dumbbell
        """
        # # array of arm joint locations for joint 0
        # arm_joint_0 = [math.pi/2, 0, -1 * math.pi/2]

        # # select location based on data direction 
        # arm_joint_0_goal = arm_joint_0[data.direction]

        # gripper_joint_close = [0.005, 0.005]

        # self.move_group_gripper.go(gripper_joint_close)
        # self.move_group_gripper.stop()

        # wait=True ensures that the movement is synchronous
        # self.move_group_arm.go([arm_joint_0_goal, 0, 0, 0], wait=True)
        self.move_group_arm.go([0.0,
                    math.radians(0.0),
                    math.radians(-40.0),
                    math.radians(-20.0)], wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group_arm.stop()


    def find_number(self, msg):
        """ Find block with target ID
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image = np.asarray(image)
        prediction_groups = self.keras_pipeline.recognize([image])
        print("Predicted groups:", prediction_groups)
        #if self.block_id in prediction_groups:
        #    self.set_v(1,0)
        #else: 
        #    self.set_v(0,.1)

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
