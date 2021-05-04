#!/usr/bin/env python3

import rospy
from std_msgs.msg import Empty

from actions import Actions

class SendActions(Actions):
    def __init__(self):
        super().__init__("q_learning", True)

        rospy.Subscriber("/q_learning/ready_for_actions", Empty, self.ready_for_actions)


    def ready_for_actions(self, data):
        while self.do_next_action():
            self.current_state = self.next_state


if __name__ == "__main__":
    node = SendActions()
    node.run()
