#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

LOOKAHEAD_WPS = 20 # Number of waypoints we will publish.


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # A list of all waypoints
        self.all_waypoints = []
        self.current_pose = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Subscribers for /traffic_waypoint and /obstacle_waypoint
        rospy.Subscriber('/traffic_waypoint', Lane, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Create waypoints Lane for publishing
        lane = Lane()
        lane.header.frame_id = '/world'

        # Start loop, 10 times a second
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            
            # Publish waypoints
            if len(self.all_waypoints) > 0 and self.current_pose is not None:

                # Get waypoints
                waypoints = self.all_waypoints

                # Find the nearest waypoint
                position = self.current_pose.position
                closest_distance = float('inf')
                closest_waypoint = 0
                for i in range(len(waypoints)):
                    this_distance = self.distance_to_position(waypoints, i, position)
                    if this_distance < closest_distance:
                        closest_distance = this_distance
                        closest_waypoint = i

                # Log
                rospy.loginfo("Closest waypoint: " + str(closest_waypoint) + " of " + str(len(waypoints)) + " at distance: " + str(closest_distance) + "\nThe waypoint:\n" + str(waypoints[closest_waypoint]) )

                # Cut this f b up
                waypoints = waypoints[closest_waypoint:closest_waypoint+10]

                # Set velocity
                for i in range(0, 10):
                    self.set_waypoint_velocity(waypoints, i, 10)
                    rospy.loginfo("Waypoint:\n" + str(waypoints[i]) + "\nOur position:\n" + str(self.current_pose))

                # Publish waypoints
                lane.header.stamp = rospy.Time.now()
                lane.waypoints = waypoints
                self.final_waypoints_pub.publish(lane)

            # Sleep
            rate.sleep()


    def pose_cb(self, current_pose):

        # Save current pose
        self.current_pose = current_pose.pose

    def waypoints_cb(self, waypoints):

        # Save given waypoints
        self.all_waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message.
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message.
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance_to_position(self, waypoints, wp, position):
        calculate_distance = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        distance = calculate_distance(waypoints[wp].pose.pose.position, position)
        return distance

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
