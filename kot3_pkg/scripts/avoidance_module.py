import rospy
from std_msgs.msg import String
from constant import *

class AvoidanceModule:
    def __init__(self) -> None:
        self.lane_signal_subscriber = rospy.Subscriber(TOPIC_NAME_LANE_DETECTION, String, self.callback)
        self.lidar_signal_subscriber = rospy.Subscriber()
        self.control_action_subscriber = rospy.Subscriber()
        self.prebuiltQTable = []

    def updateEnvironmentSignal(self):
        """
        Input: Signal from all subscribers
        Ouput: ratio_left, lidar, alpha
        """
        pass

    def convertSignalToState(self, updatedSignal):
        """
        Input: ratio_left, lidar, alpha
        Ouput: ABCDE
            A (0-1): Vật thể vùng 1
            B (0-1): Vật thể vùng 2
            C (0-1): Vật thể vùng 3
            D(0-9): Hướng di chuyển của robot, vì robot chỉ quay một góc 18 độ mỗi lần, nên trong phạm vi góc -90 đến 90, ta có 10 giá trị góc mà robot có thể tồn tại.
            E(0-9): Đại diện cho vị trí robot so với chiều ngang của vùng xanh lá. Ta chia vùng xanh lá ra thành 10 vùng theo chiều dọc và đánh số từ 0 đến 9 từ trái qua phải
            Tổng cộng, ta sẽ có tất cả 800 state và 4 action
        """
        pass

    def decideActionBaseOnCurrentState(self, action):
        """
        Input: ABCDE
        Ouput: action
        """
        pass

    def sendActionToTopic(self, action):
        """
        Input: action
        Ouput: None
        """
        pass

    def callback(self, data):
        updatedSignal = self.updateEnvironmentSignal()
        state = self.convertSignalToState(updatedSignal)
        action = self.decideActionBaseOnCurrentState(state)
        self.sendActionToTopic(action)
        rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data) 

if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_AVOIDANCE, anonymous=True)
        avoidance = AvoidanceModule()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass