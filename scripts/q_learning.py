#!/usr/bin/env python3

import rospy
import numpy as np
import os
import csv

from std_msgs.msg import Header
from q_learning_project.msg import QLearningReward, QMatrix, QMatrixRow, RobotMoveDBToBlock

import cv2, cv_bridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

class QLearning(object):
    def __init__(self):
        # Initialize this node
        rospy.init_node("q_learning")

        # Fetch pre-built action matrix. This is a 2d numpy array where row indexes
        # correspond to the starting state and column indexes are the next states.
        #
        # A value of -1 indicates that it is not possible to get to the next state
        # from the starting state. Values 0-9 correspond to what action is needed
        # to go to the next state.
        #
        # e.g. self.action_matrix[0][12] = 5
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt")

        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { dumbbell: "red", block: 1}
        colors = ["red", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(
            lambda x: {"dumbbell": colors[int(x[0])], "block": int(x[1])},
            self.actions
        ))


        # Fetch states. There are 64 states. Each row index corresponds to the
        # state number, and the value is a list of 3 items indicating the positions
        # of the red, green, blue dumbbells respectively.
        # e.g. [[0, 0, 0], [1, 0 , 0], [2, 0, 0], ..., [3, 3, 3]]
        # e.g. [0, 1, 2] indicates that the green dumbbell is at block 1, and blue at block 2.
        # A value of 0 corresponds to the origin. 1/2/3 corresponds to the block number.
        # Note: that not all states are possible to get to.
        self.states = np.loadtxt(path_prefix + "states.txt")
        self.states = list(map(lambda x: list(map(lambda y: int(y), x)), self.states))

        rospy.Subscriber("/q_learning/reward", QLearningReward, self.get_learning_reward)
        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()
        # initalize the debugging window
        cv2.namedWindow("window", 1)
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
                        Image, self.find_dumbbells)

        self.matrix_pub = rospy.Publisher("/q_learning/q_matrix", QMatrix, queue_size=10)
        self.action_pub = rospy.Publisher("/q_learning/robot_action", RobotMoveDBToBlock, queue_size=10)

        rospy.sleep(2)

        self.use_saved_matrix = False
        self.q_matrix_path = 'output/q_matrix.csv'

        self.q_matrix = []
        self.init_q_matrix()

        self.unchanged_count = 0

        self.current_state = 0
        self.next_state = None
        self.action_num = None
        self.do_next_action()

        self.dumbbell_target = 0


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

    def init_q_matrix(self):
        if self.use_saved_matrix:
            self.load_q_matrix()
            return

        for i in range(64):
            row = list(np.repeat(0, 9))
            self.q_matrix.append(row)


    def publish_q_matrix(self):
        m = QMatrix()
        l = []
        for row in self.q_matrix:
            r = QMatrixRow()
            r.q_matrix_row = row
            l.append(r)

        m.header = Header(stamp=rospy.Time.now())
        m.q_matrix = l
        self.matrix_pub.publish(m)


    def choose_action(self, action_options):
        allowed_action_idxs = []
        for i in range(len(action_options)):
            action = int(action_options[i])
            if action != -1:
                allowed_action_idxs.append(i)

        if len(allowed_action_idxs) == 0:
            return None
        
        if self.use_saved_matrix:
            best_q_value = None
            best_action_num = None
            best_next_state = None
            for action_idx in allowed_action_idxs:
                action_num = int(action_options[action_idx])
                next_state = action_idx
                q_value = self.q_matrix[self.current_state][action_num]
                if best_q_value is None or q_value > best_q_value:
                    best_q_value = q_value
                    best_action_num = action_num
                    best_next_state = next_state
            return (best_action_num, best_next_state)

        action_idx = np.random.choice(allowed_action_idxs)
        action_num = int(action_options[action_idx])
        next_state = action_idx
        return (action_num, next_state)


    def do_next_action(self):
        action_options = self.action_matrix[self.current_state]
        res = self.choose_action(action_options)
        if res is None:
            if self.use_saved_matrix:
                return

            self.current_state = 0
            return self.do_next_action()
        action_num = res[0]
        next_state = res[1]
        
        action_data = self.actions[action_num]
        action = RobotMoveDBToBlock()
        action.robot_db = action_data['dumbbell']
        action.block_id = action_data['block']

        self.action_num = action_num
        self.next_state = next_state
        self.action_pub.publish(action)


    def find_max_reward(self, row):
        return max(row)


    def get_learning_reward(self, data):
        reward = data.reward
        next_r = self.find_max_reward(self.q_matrix[self.next_state])

        current_val = self.q_matrix[self.current_state][self.action_num]
        new_val = int(round(reward + 0.8 * next_r))
        if current_val == new_val:
            self.unchanged_count += 1
        else:
            self.unchanged_count = 0
            self.q_matrix[self.current_state][self.action_num] = new_val
            self.publish_q_matrix()

        if self.unchanged_count == 100:
            self.save_q_matrix()
            return
        
        self.current_state = self.next_state
        self.do_next_action()


    def save_q_matrix(self):
        with open(self.q_matrix_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.q_matrix:
                writer.writerow(row)


    def load_q_matrix(self):
        with open(self.q_matrix_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.q_matrix.append(list(map(lambda x: int(x), row)))


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = QLearning()
    rospy.spin()
