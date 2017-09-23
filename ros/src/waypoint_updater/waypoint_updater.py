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

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.waypoints = None

        rospy.spin()

    def pose_cb(self, msg):
        if self.waypoints:
            position = msg.pose

            theta = math.asin(position.orientation.z) * 2
            current_x = position.position.x
            current_y = position.position.y
            # simulating a position just ahead of the vehicle
            future_x = math.cos(theta) * 0.0001 + current_x
            future_y = math.sin(theta) * 0.0001 + current_y

            closest_index = 0
            closest_waypoint = None
            closest_distance = 1e99
            for i, waypoint in enumerate(self.waypoints.waypoints):
                this_x = waypoint.pose.pose.position.x
                this_y = waypoint.pose.pose.position.y
                this_distance = self.orthogonal_distance(this_x, this_y, current_x, current_y)
                if this_distance < closest_distance:
                    closest_index = i
                    closest_waypoint = waypoint
                    closest_distance = this_distance
            output = Lane()
            output.header = self.waypoints.header
            num_waypoints = len(self.waypoints.waypoints)
            if closest_distance > self.orthogonal_distance(closest_waypoint.pose.pose.position.x,
                                                           closest_waypoint.pose.pose.position.y,
                                                           future_x, future_y):
                output.waypoints = self.waypoints.waypoints[
                                   closest_index:(closest_index + LOOKAHEAD_WPS) % num_waypoints]
            else: # closest waypoint is behind, so use next waypoint
                output.waypoints = self.waypoints.waypoints[
                                   (closest_index + 1) % num_waypoints :(closest_index + LOOKAHEAD_WPS + 1) %
                                                                       num_waypoints]
            self.final_waypoints_pub.publish(output)
        else:
            rospy.logwarn("Original waypoints not yet loaded. Cannot publish final waypoints.")

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
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

    def orthogonal_distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')