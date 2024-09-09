# Import necessary libraries
import rosbag
import message_filters
from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# Global variables
drive = '/vesc/low_level/ackermann_cmd_mux/input/teleop'  # Topic for drive commands
joy = '/vesc/joy'  # Topic for joystick commands
lid = '/scan_filtered'  # Lidar topic (filtered)
pressed = False  # Button press flag
bag = None  # ROS Bag to store collected data
bag_name = 'Dataset/out.bag' # path and name of the bag where the data needs to be stored

# ROS initialization
rospy.init_node('receive_position')

# Callback for receiving Ackermann and Lidar messages
def callback(ack_msg, ldr_msg):
    if pressed and bag is not None:
        print(f'Ackermann: {ack_msg}')
        print(f'Lidar: {ldr_msg}')
        bag.write('Ackermann', ack_msg)  # Write Ackermann messages to bag
        bag.write('Lidar', ldr_msg)  # Write Lidar messages to bag

# Callback for receiving button press from joystick
def button_callback(j):
    global pressed
    global bag

    # Check if button 1 is pressed which is mapped to button A of Logistic Joystick
    if j.buttons[1] == 1 and not pressed:
        pressed = True
        bag = rosbag.Bag(bag_name, 'w')  # Create a new bag for recording
        print('Recording Started')
    elif j.buttons[1] == 0 and pressed:
        pressed = False
        bag.close()  # Close the bag when recording stops
        print('Recording Stopped')

# Create subscribers for drive and lidar messages
drive_sub = message_filters.Subscriber(drive, AckermannDriveStamped)
lid_sub = message_filters.Subscriber(lid, LaserScan)

# Use time synchronizer to combine messages from both topics
ts = message_filters.ApproximateTimeSynchronizer([drive_sub, lid_sub], queue_size=10, slop=0.01, allow_headerless=True)

# Subscribe to joystick topic to receive button press
rospy.Subscriber(joy, Joy, button_callback)

# Register the callback function
ts.registerCallback(callback)

# Keep the program running
rospy.spin()

# End of data collection
print('\n-----------Recording Completed-----------')