#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import moveit_commander
import math
import statistics
import time
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
        # set self.front_distance to distance to nearest object in front of robot
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.get_distance)
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.complete_action)
        self.ready_pub = rospy.Publisher("/q_learning/ready_for_actions", Empty, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.action_queue = []
        self.front_distance = 100
        self.front_left = 100
        self.front_right = 100
        self.carrying_db = False
        self.carrying_db_successful = False
        rospy.Subscriber("/q_learning/robot_action", RobotMoveDBToBlock, self.handle_robot_action)
        rospy.sleep(1)
        self.ready_pub.publish(Empty())

        self.turning = False
        self.turn_1 = False
        self.arm_initialized = False

        self.drive_to_block = False

        self.past_counter = 0

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        print("Initialized")
        self.initialized = True


    def handle_robot_action(self, data):
        # when we get a new action, we simply queue it and handle it later
        self.action_queue.append(data)


    def get_distance(self, msg):
        # use a few front distances to handle when we are nearing objects
        self.front_distance = msg.ranges[0]
        self.front_left = msg.ranges[330]
        self.front_right = msg.ranges[30]


    def complete_action(self, msg):
        """ This is the callback for the image sensor reading, where we do all of the
        robot action handling
        """
        if (not self.initialized):
            return

        if len(self.action_queue) == 0:
            return
        target = self.action_queue[0]
        color = target.robot_db # "red", "blue", "green"
        self.block_id = target.block_id # 1, 2, 3

        self.dumbbell_target = np.where(np.asarray(["red","green","blue"])==color)[0][0]

        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, w, d = image.shape

        # if we are currently carrying a dumbbell, we first determine where the desired block
        # is, drive towards it, then place the dumbbell down when we are close enough
        if self.carrying_db:
            if self.drive_to_block:
                if self.front_distance > .75 and self.front_left > .75 and self.front_right > .75:
                    self.set_v(.2, 0)
                else:
                    self.set_db()
                    # remove the current action from the queue and reset everything so that
                    # we can start looking for the next dumbbell
                    self.action_queue.pop(0)
                    self.carrying_db = False
                    self.drive_to_block = False
                    self.turning = False

                return

            # here we turn and run image recognition to look for our target number at
            # frequent intervals
            if self.turning:
                self.set_v(0, .5)
                self.turning = False
                if self.turn_1:
                    rospy.sleep(4)
                    self.turn_1 = False
                else:
                    rospy.sleep(2)
                self.set_v(0, 0)
                rospy.sleep(2)
            else:
                # when there is a long delay since the previous image reading that we handled,
                # we need to iterate through some image sensor readings until we get to the most
                # recent one because it seems that ROS sends some historical image sensor readings
                # before sending along an up-to-date reading
                if self.past_counter < 100:
                    self.past_counter += 1
                    return
                self.past_counter = 0
                pos = self.find_number(msg)
                print("POS:",pos)
                if pos is None:
                    # the target number is not in the robot scan so we resume turning
                    self.turning = True
                else:
                    block_err = w/2 - pos
                    print('Block Error:', block_err)
                    k_p = .0025
                    scaled = k_p*block_err
                    print('Error:',scaled)
                    if abs(block_err) > 75:
                        # use this averaged target number to try and slowly center the robot on the image
                        sign = 1 if scaled >= 0 else -1
                        angular_vel = min(abs(scaled), .5)
                        self.set_v(0., sign * angular_vel)
                        rospy.sleep(.5)
                        self.set_v(0, 0)
                        rospy.sleep(.5)
                    else:
                        self.drive_to_block = True

                # cv2.imshow("window", image)
                # cv2.waitKey(3)
            return

        if not self.arm_initialized:
            self.initialize_arm()
            self.arm_initialized = True

        # define the upper and lower bounds rgb
        lower_bounds = np.array([0, 121/2, 241/2]) 
        upper_bounds = np.array([20, 180/2, 300/2])
        rgb_lower = [np.asarray([lower_bounds[i],20, 20]) for i in range(3)]
        rgb_upper = [np.asarray([upper_bounds[i],255, 255]) for i in range(3)]
        
        mask = cv2.inRange(hsv, rgb_lower[self.dumbbell_target], rgb_upper[self.dumbbell_target])
        if color == "red": 
            mask2 = cv2.inRange(hsv, np.asarray([170/2, 20, 20]), np.asarray([180/2, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)

        # using moments() function, the center of the colored pixels is determined
        M = cv2.moments(mask)
        # if there are any colored pixels found
        if M['m00'] > 0:
                # center of the colored pixels in the image
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                err = w/2 - cx
                k_p = .003
                # drive towards the dumbbell until we are close enough to pick it up
                if self.front_distance > .21:
                    if abs(err) > .05: 
                        self.set_v(.05, k_p*err)
                    else:
                        self.set_v(.13, k_p*err)
                else:
                    self.set_v(0,0)
                    #pick up dumbbell
                    self.pickup_db()
                    self.carrying_db = True
                    self.turn_1 = True
                    self.turning = True
        else:
            # spin if the goal color is not visible
            self.set_v(0, .4)


    def initialize_arm(self):
        gripper_joint_open = [0.01, 0.01]
        self.move_group_gripper.go(gripper_joint_open, wait=True)
        self.move_group_gripper.stop()

        # wait=True ensures that the movement is synchronous
        self.move_group_arm.go([0.0,
                    math.radians(50.0),
                    math.radians(-35.0),
                    math.radians(-15.0)], wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group_arm.stop()


    def set_db(self):
        """ set down the dumbbell with the gripper then move backwards away from it
        """
        self.set_v(0, 0)
        rospy.sleep(1)

        rospy.sleep(1)

        self.move_group_arm.go([0.0,
                    math.radians(45.0),
                    math.radians(-20.0),
                    math.radians(-10.0)], wait=True)
        self.move_group_arm.stop()

        rospy.sleep(1)

        gripper_joint_open = [0.01, 0.01]
        self.move_group_gripper.go(gripper_joint_open, wait=True)
        self.move_group_gripper.stop()

        self.set_v(-.5, 0)
        rospy.sleep(1)

        self.initialize_arm()

        self.set_v(0, 0)

        rospy.sleep(1)


    def pickup_db(self):
        """ Pickup dumbbell
        """
        self.move_group_arm.go([0.0,
                    math.radians(0.0),
                    math.radians(-40.0),
                    math.radians(-20.0)], wait=True)
        self.move_group_arm.stop()


    def find_number(self, msg):
        """ Find block with target ID and return the x-coord of the center of the target
        number in the image, if the target number is present. Otherwise return None
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        prediction_groups = self.keras_pipeline.recognize([image])
        print("Predicted groups:", prediction_groups)

        block_num = self.block_id
        print(f'Checking for block_id {block_num}')
        centers = []
        for g in prediction_groups[0]:
            corners = g[1]
            if g[0] == str(block_num):
                c = (corners[0][0] + corners[1][0]) / 2
                centers.append(c)
        
        if len(centers) == 0:
            return None
        if block_num == 3 and len(centers) == 2:
            center = centers[1]
        elif block_num == 1 and len(centers) == 2: 
            center = centers[0]
        else: 
            center = centers[0]
        return center


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
