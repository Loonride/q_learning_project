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

class Actions(object):
    """ This is a base class that the QLearning and SendActions classes inherit from.
    This setup is useful because both training and action sending requires knowledge
    of the action matrix.
    """

    def __init__(self, node_name, use_saved_matrix):
        """
        Parameters:
        node_name
            set the rospy node name
        use_saved_matrix
            Determines whether the q-matrix should be initialized empty or if it should
            load from the q-matrix output file. This should be False if we are training
            and True if we are running actions based on the saved q-matrix
        """
        # Initialize this node
        rospy.init_node(node_name)

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

        self.matrix_pub = rospy.Publisher("/q_learning/q_matrix", QMatrix, queue_size=10)
        self.action_pub = rospy.Publisher("/q_learning/robot_action", RobotMoveDBToBlock, queue_size=10)

        # 
        self.use_saved_matrix = use_saved_matrix

        dirname = os.path.dirname(__file__)
        self.q_matrix_path = os.path.join(dirname, '../output/q_matrix.csv')

        self.q_matrix = []
        self.init_q_matrix()

        self.unchanged_count = 0

        self.current_state = 0
        self.next_state = None
        self.action_num = None


    def init_q_matrix(self):
        # if using the saved output matrix, we should load it rather than
        # initializing with zeros
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
        """ Choose the best action given the list of states that can be transitioned to
        from the current state (action_options)
        """
        allowed_action_idxs = []
        for i in range(len(action_options)):
            action = int(action_options[i])
            if action != -1:
                allowed_action_idxs.append(i)

        if len(allowed_action_idxs) == 0:
            return None
        
        # if we are choosing actions based on the saved matrix, we should select
        # the action with the best weight at the current state in the q-matrix
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

        # during training, we should randomly select from the allowed actions and
        # transition to the next state to continue training
        action_idx = np.random.choice(allowed_action_idxs)
        action_num = int(action_options[action_idx])
        next_state = action_idx
        return (action_num, next_state)


    def do_next_action(self):
        action_options = self.action_matrix[self.current_state]
        res = self.choose_action(action_options)
        # this indicates there was no state to transition to, so we either quit
        # or continue training after a reset
        if res is None:
            if self.use_saved_matrix:
                return False

            self.current_state = 0
            self.do_next_action()
            return True
        action_num = res[0]
        next_state = res[1]
        
        # publish the chosen action
        action_data = self.actions[action_num]
        action = RobotMoveDBToBlock()
        action.robot_db = action_data['dumbbell']
        action.block_id = action_data['block']

        self.action_num = action_num
        self.next_state = next_state
        self.action_pub.publish(action)
        return True


    def save_q_matrix(self):
        # write q-matrix to a csv file
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
