#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from tf.transformations import euler_from_quaternion

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

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704

class WaypointUpdater(object):
    def __init__(self):
        """
        Initialize the node, states and params. Set callbacks for topics.
        """
        rospy.init_node('waypoint_updater')
        rospy.loginfo("WaypointUpdater: Initialized.")

        self.curr_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # initialize states and params
        self.base_waypoints = None
        self.target_velocity = 10.0
        self.last_waypoint = None
        self.pose_received_time = None
        self.traffic_received_time = None

        rospy.spin()

    def __is_behind(self, pose, target_wp, closest_distance):
        """
        To determine if the target waypoint is behind the car
        :param pose: the pose of the car
        :param target_wp: the target waypoint
        :return: bool True if waypoint is behind else False
        """

        theta = math.asin(pose.orientation.z) * 2
        current_x = pose.position.x
        current_y = pose.position.y

        # simulate a position just ahead of the vehicle
        future_x = math.cos(theta) * 0.0001 + current_x
        future_y = math.sin(theta) * 0.0001 + current_y

        # if closest distance is less than distance between future position and target waypoint
        if closest_distance <= self.euclidean_distance(target_wp.pose.pose.position.x,
                                                       target_wp.pose.pose.position.y,
                                                       future_x,
                                                       future_y):
            return True

        return False

    def __get_closest_waypoint(self, pose):
        """
        Get the closest waypoint from the car's current pose
        :param pose: the pose of the car
        :return: the closest waypoint ahead of the car
        """

        # get the current position
        current_x = pose.position.x
        current_y = pose.position.y
        num_waypoints = len(self.base_waypoints)

        # calculate the nearest waypoint by temporarily storing shortest distance
        closest_index = 0
        closest_waypoint = None
        closest_distance = float('inf')

        # check if we have a known last point; if found lopp from it to improve performance
        if self.last_waypoint:  # will happen every time except for the first time.
            for i in range(num_waypoints):
                this_index = (i + self.last_waypoint) % num_waypoints
                waypoint = self.base_waypoints[this_index]
                this_x = waypoint.pose.pose.position.x
                this_y = waypoint.pose.pose.position.y
                this_distance = self.euclidean_distance(this_x, this_y, current_x, current_y)
                if this_distance < closest_distance:
                    closest_index = this_index
                    closest_waypoint = waypoint
                    closest_distance = this_distance
                else:  # we are now getting further away, stop
                    break
        # iterate through all waypoints initially
        else:
            for i, waypoint in enumerate(self.base_waypoints):
                this_x = waypoint.pose.pose.position.x
                this_y = waypoint.pose.pose.position.y
                this_distance = self.euclidean_distance(this_x, this_y, current_x, current_y)
                if this_distance < closest_distance:
                    closest_index = i
                    closest_waypoint = waypoint
                    closest_distance = this_distance

        # check if closest waypoint is behind the car
        if self.__is_behind(pose, closest_waypoint, closest_distance):
            closest_index += 1

        return closest_index

    def __generate_next_waypoints(self, index):
        """
        To generate new waypoints from the car's current pose
        :param index: the closest waypoint index
        :return: a list of waypoints ahead of the car
        """
        new_waypoints = []
        for i in range(LOOKAHEAD_WPS):
            new_waypoints.append(self.base_waypoints[(i + index) % len(self.base_waypoints)])

        return new_waypoints

    def pose_cb(self, msg):
        """
        Callback to handle pose updates. Generates the next waypoints
        :param msg: incoming message, contains pose data
        :return: None
        """
        self.pose_received_time = rospy.Time.now()
        if self.base_waypoints is not None:

            pose = msg.pose
            closest_index = self.__get_closest_waypoint(pose)
            next_waypoints = self.__generate_next_waypoints(closest_index)

            rospy.loginfo("Default velocity is {} ".format(next_waypoints[0].twist.twist.linear.x))
            # TODO: smoothen the speed across waypoints; check if traffic light is detected
            for wp in next_waypoints:
                wp.twist.twist.linear.x = self.target_velocity * ONE_MPH

            # get the lane object
            lane = self.__get_lane(msg.header, next_waypoints)

            # publish the waypoints
            self.final_waypoints_pub.publish(lane)
            self.last_waypoint = closest_index
            rospy.loginfo("Published {} new waypoints".format(len(next_waypoints)))
        else:
            rospy.logwarn("Original waypoints not yet loaded. Cannot publish final waypoints.")


    def __get_lane(self, header, waypoints):
        """
        To get the lane object for publishing
        :param header: the message header
        :param waypoints: the new waypoints
        :return: the Lane object
        """
        lane = Lane()
        lane.header = header
        lane.waypoints = waypoints

        return lane

    def waypoints_cb(self, msg):
        """
        Callback to handle incoming waypoints message.
        :param msg: incoming message, contains waypoints
        :return: None
        """
        self.base_waypoints = msg.waypoints

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
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def euclidean_distance(self, x1, y1, x2, y2):
        """
        To calculate euclidean distance between two poses
        :param x1: the x position of first pose
        :param y1: the y position of first pose
        :param x2: the x position of second pose
        :param y2: the y position of second pose
        :return: the distance between the two poses
        """
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')