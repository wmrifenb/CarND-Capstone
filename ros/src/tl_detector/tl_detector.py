#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        
        self.pose = None
        self.waypoints = []
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub4 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # The publisher of the next red traffic light waypoint index
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # Find the nearest waypoint
        closest_distance = float('inf')
        closest_waypoint = 0
        for i in range(len(self.waypoints)):
            this_distance = self.distance_to_position(self.waypoints, i, pose.position)
            if this_distance < closest_distance:
                closest_distance = this_distance
                closest_waypoint = i
        return closest_waypoint
        
    def get_closest_stop_line_waypoint(self, waypoints, car_position_waypoint, stop_line_positions):
    
        # Go through all stop_lines, and find their nearest waypoint
        stop_line_waypoints = []
        for stop_line_position in stop_line_positions:

            # Get position of this stop line
            pose = Pose()
            pose.position.x = stop_line_position[0]
            pose.position.y = stop_line_position[1]
            
            # Find the nearest waypoint to this stopline position
            closest_distance = float('inf')
            closest_waypoint = 0
            for i, waypoint in enumerate(self.waypoints):
                this_distance = self.distance_to_position(self.waypoints, i, pose.position)
                if this_distance < closest_distance:
                    closest_distance = this_distance
                    closest_waypoint = i
            stop_line_waypoints.append(closest_waypoint)

        # Log
        #rospy.loginfo("stop_line_waypoints:")
        #rospy.loginfo(stop_line_waypoints)

        # Now find the stop_line waypoint that closest to but not behind car_position_waypoint
        closest_stop_line_waypoint = 1000000
        closest_stop_line_index = -1
        for i, stop_line_waypoint in enumerate(stop_line_waypoints):
            # If it's past the car, and closer than we've seen so far
            if stop_line_waypoint > car_position_waypoint and stop_line_waypoint < closest_stop_line_waypoint:
                closest_stop_line_waypoint = stop_line_waypoint
                closest_stop_line_index = i

        # Return
        return closest_stop_line_index, closest_stop_line_waypoint

    def distance_to_position(self, waypoints, wp, position):
        calculate_distance = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        distance = calculate_distance(waypoints[wp].pose.pose.position, position)
        return distance

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # Our closest visible traffic light starts off as none
        closest_traffic_light_waypoint = -1

        # Get the list of 'stop line' positions
        stop_line_positions = self.config['stop_line_positions']
        
        # Find which waypoint the car is closest to
        closest_stop_line_waypoint = -1
        index = -1
        traffic_light_state = TrafficLight.UNKNOWN
        if(self.pose):
            car_position_waypoint = self.get_closest_waypoint(self.pose.pose)

            # Find the closest traffic light based on stop positions
            index, closest_stop_line_waypoint = self.get_closest_stop_line_waypoint(self.waypoints, car_position_waypoint, stop_line_positions)

            # Get traffic light state
            if len(self.lights) > 0 and index >= 0 and index < len(self.lights):
                # Use matching index from stop_line_positions and hope that they align
                traffic_light_state = self.lights[index].state

            # Log
            rospy.loginfo("car_position_waypoint: " + str(car_position_waypoint))
            rospy.loginfo("closest_stop_line_waypoint: " + str(closest_stop_line_waypoint))
    #        rospy.loginfo("stop_line_positions:")  
    #        rospy.loginfo(stop_line_positions)
            rospy.loginfo("traffic light state: " + str(self.lights[0].state))
        
        return closest_stop_line_waypoint, traffic_light_state

#        if light:
#            state = self.get_light_state(light)
#            return light_wp, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
