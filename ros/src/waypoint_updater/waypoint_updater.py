#!/usr/bin/env python

"""The waypoint updater node."""

__author__ = "Thomas Woodside, Karthick Loganathan"

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
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

LOOKAHEAD_WPS = 150  # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704
WAIT_TIME = 10.0
SAFE_ACCEL = 5.0

class WaypointUpdater(object):
    def __init__(self):
        """
        Initialize the node, states and params. Set callbacks for topics.
        """
        rospy.init_node('waypoint_updater')
        rospy.loginfo("WaypointUpdater: Initialized.")

        # register the subscribers
        self.curr_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.traffic_sub = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.curr_vel_sub = rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)

        # create the publisher
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # initialize states and params
        self.base_waypoints = None
        self.max_speed = rospy.get_param('~velocity', 10.0)
        self.last_waypoint = None
        self.pose_received_time = None
        self.traffic_received_time = None
        self.traffic_wp_ind = None
        self.current_speed = 0.0

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

    def __generate_next_waypoints(self, start_index, last_index):
        """
        To generate new waypoints from the car's current pose
        :param start_index: the closest waypoint index
        :param last_index: the final waypoint index
        :return: a list of waypoints ahead of the car
        """
        new_waypoints = []
        i = 0
        while True:
            new_index = (i + start_index) % len(self.base_waypoints)
            new_waypoints.append(self.base_waypoints[new_index])
            i = i+1
            if new_index == last_index:
                break

        return new_waypoints

    def __get_traffic_wp(self, closest_index):
        '''
        To get a valid traffic waypoint index ahead of the car
        :param closest_index: the index of the waypoint near the car
        :return: the traffic waypoint index; -1 if no traffic waypoints ahead
        '''

        # check if we received a traffic waypoint index
        if self.traffic_wp_ind is not None:
            # check if the traffic index is still valid and not old
            if (rospy.Time.now().to_sec() - self.traffic_received_time) <= WAIT_TIME:
                # check if the traffic index is ahead of the planned route
                for i in range(LOOKAHEAD_WPS):
                    new_index = (i + closest_index) % len(self.base_waypoints)
                    if new_index == self.traffic_wp_ind:
                        return self.traffic_wp_ind
            # if traffic index is old reset the waypoint params
            else:
                self.traffic_wp_ind = None
                self.traffic_received_time = None

        # not found
        return -1

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
            last_index = (closest_index + LOOKAHEAD_WPS) % len(self.base_waypoints) - 1

            # get the traffic waypoint index
            traffic_wp_ind = self.__get_traffic_wp(closest_index)

            # check if car has to brake and also get the desired speed
            brake = False
            max_speed = self.max_speed * ONE_MPH
            if traffic_wp_ind != -1:
                brake = True
                max_speed = 0.0
                last_index = traffic_wp_ind

            # generate the next set of waypoints
            next_waypoints = self.__generate_next_waypoints(closest_index, last_index)

            # set the speed for the waypoints
            next_waypoints = self.__set_speed(next_waypoints, max_speed, brake)

            # get the lane object
            lane = self.__get_lane(msg.header, next_waypoints)

            # publish the waypoints
            self.final_waypoints_pub.publish(lane)
            self.last_waypoint = closest_index
        else:
            rospy.logwarn("Original waypoints not yet loaded. Cannot publish final waypoints.")

    def __set_speed(self, waypoints, final_speed, brake=False):
        """
        To set the desired speed for the waypoints
        :param waypoints: the list of target waypoints
        :param final_speed: the desired speed
        :param brake: indicates if the car has to stop
        :return: the list of target waypoints with smooth speed
        """

        if waypoints is None:
            return waypoints

        # get the current speed of the vehicle
        current_speed = self.current_speed

        # increase the speed slowly to reach the target speed
        if not brake:
            for wp in waypoints:
                current_speed = min(current_speed + SAFE_ACCEL, final_speed)
                self.set_waypoint_velocity(wp, current_speed)
        else:
            # if the car is stationary; start slowly so the car stops at the stop line and not before
            if current_speed <0.1 and len(waypoints>2):
                current_speed = SAFE_ACCEL
            # reduce the speed slowly based on the number of waypoints
            num_wps = len(waypoints)
            brake_slowdown = math.fabs((final_speed - current_speed)) / num_wps
            for wp in waypoints:
                current_speed = max(current_speed - (brake_slowdown), 0.0)
                self.set_waypoint_velocity(wp, current_speed)
            # set the last waypoint to 0 just in case there is a residue
            waypoints[-1].twist.twist.linear.x = 0.0

        return waypoints

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
        '''
        Callback to handle incoming traffic waypoint messages
        :param msg: the incoming message containing traffic waypoint data
        :return: None
        '''
        self.traffic_wp_ind = msg.data
        # set received time so we will know when the car can move
        self.traffic_received_time = rospy.Time.now().to_sec()

    def velocity_cb(self, msg):
        """
        Callback to set the current speed of the car
        :param msg: incoming message containing current velocity
        :return: None
        """
        # rospy.loginfo("WaypointUpdater: Received current speed {}".format(msg.twist.linear.x))
        self.current_speed = msg.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoint, velocity):
        """
        Sets a the desired velocity for the waypoint. Thresholds the max value to the speed limit
        :param waypoint: the waypoint to set the speed for
        :param velocity: the desired speed
        :return: None
        """
        safe_speed = min(velocity, self.max_speed * ONE_MPH)
        waypoint.twist.twist.linear.x = safe_speed


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