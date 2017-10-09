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
            if len(self.all_waypoints) > LOOKAHEAD_WPS:
                # Publish the first LOOKAHEAD_WPS waypoints
                waypoints = self.all_waypoints[0:LOOKAHEAD_WPS]
                lane.header.stamp = rospy.Time.now()
                lane.waypoints = waypoints
                self.final_waypoints_pub.publish(lane)

            # Sleep
            rate.sleep()


    def pose_cb(self, msg):
        # TODO: Implement
        pass

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
