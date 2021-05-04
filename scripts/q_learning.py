#!/usr/bin/env python3

import rospy
from q_learning_project.msg import QLearningReward

from actions import Actions

class QLearning(Actions):
    def __init__(self):
        super().__init__("q_learning", False)

        rospy.Subscriber("/q_learning/reward", QLearningReward, self.get_learning_reward)

        rospy.sleep(2)
        self.do_next_action()


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

        if self.unchanged_count == 500:
            self.save_q_matrix()
            print('Matrix Saved')
            return
        
        self.current_state = self.next_state
        self.do_next_action()


if __name__ == "__main__":
    node = QLearning()
    node.run()
