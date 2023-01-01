import rospy
from std_msgs.msg import String
from constant import *

class AvoidanceModule:
    def __init__(self) -> None:
        self.lane_signal_subscriber = rospy.Subscriber(TOPIC_NAME_LANE_DETECTION, String, self.callback)
        self.lidar_signal_subscriber = rospy.Subscriber()
        self.prebuiltQTable = []

    def convertSignalToState(self):
        pass

    def decideActionBaseOnCurrentState(action):
        pass

    def callback(self, data):
        state = self.convertSignalToState()
        action = self.decideActionBaseOnCurrentState(state)
        rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)    



if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_AVOIDANCE)
        avoidance = AvoidanceModule()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass