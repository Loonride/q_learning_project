#!/usr/bin/env python3

import rospy
from std_msgs.msg import Empty

from actions import Actions

class SendActions(Actions):
    """ This class inherits from the base class Actions, which interacts with the action
    matrix and has helpers to choose next actions
    """
    def __init__(self):
        super().__init__("q_learning", True)

        # we wait until the movement node is ready to get actions before we start
        # sending them
        rospy.Subscriber("/q_learning/ready_for_actions", Empty, self.ready_for_actions)


    def ready_for_actions(self, data):
        while self.do_next_action():
            self.current_state = self.next_state


if __name__ == "__main__":
    node = SendActions()
    node.run()
