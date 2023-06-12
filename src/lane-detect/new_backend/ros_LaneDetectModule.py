import rospy
from std_msgs.msg import String

pub = 0

def callback(data):
    
    x=msg.linear.x
    y=msg.linear.y
    z=msg.angular.z

    rospy.loginfo("I got ",x,y,z)
    pub.publish(data.data)
    
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('processNode', anonymous=True)

    rospy.Subscriber("cmd_vel", String, callback)
    pub = rospy.Publisher('somethingElse', String, queue_size=10)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

