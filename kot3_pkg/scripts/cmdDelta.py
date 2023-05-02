#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

class MoveForward:
    def __init__(self):
        rospy.init_node('move_forward', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.move_msg = Twist()
        # self.move_msg.angular.z = 0.2 
        self.move_msg.linear.x = -0.05 
        self.stop = Twist()
        self.b = True

    def move(self):
        rate = rospy.Rate(5)
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 10.0: # move for 1 second
            if self.b:
                self.velocity_publisher.publish(self.move_msg)
            else:
                self.velocity_publisher.publish(self.stop)
            self.b = not self.b
            rate.sleep()
        print("total time:", (rospy.Time.now() - start_time).to_sec())
        self.velocity_publisher.publish(self.stop)

if __name__ == '__main__':
    try:
        print("init")
        robot = MoveForward()
        print("move")
        robot.move()
        print("done")
    except rospy.ROSInterruptException:
        pass