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

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        rospy.loginfo("WaypointUpdater: Initialized.")

        self.curr_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # initialize states
        self.base_waypoints = None
        self.pose = None

        # set desired params
        self.target_velocity = 10.0

        rospy.spin()

    def __get_distance(self, pose1, pose2):
        '''
        To calculate euclidean distance between two poses
        :param pose1: the first pose
        :param pose2: the second pose
        :return: the distance between the two poses
        '''
        return math.sqrt(
            math.pow(pose1.position.x - pose2.position.x, 2) + math.pow(pose1.position.y - pose2.position.y, 2))

    def __is_behind(self, origin_pose, target_wp):
        '''
        To determine if the target waypoint is behind the car
        :param origin_pose: the pose of the car
        :param target_wp: the target waypoint
        :return: bool True if waypoint is behind else False
        '''

        # get the orientation angle from the quaternion pose
        _, _, yaw = euler_from_quaternion(
            [origin_pose.orientation.x, origin_pose.orientation.y, origin_pose.orientation.z,
             origin_pose.orientation.w])

        # shift the co-ordinates to be in a straight line
        shift_x = target_wp.position.x - origin_pose.position.x
        shift_y = target_wp.position.y - origin_pose.position.y

        # rotate the yaw to determine if it is positive
        x = shift_x * math.cos(-yaw) - shift_y* math.sin(-yaw)
        if x > 0.0:
            return False

        # return True if angle is negative; indicates the waypoint is behind
        return True


    def __get_closest_waypoint(self, pose):
        '''
        Get the closest waypoint from the car's current pose
        :param pose: the pose of the car
        :return: the closest waypoint ahead of the car
        '''

        closest_distance = float('inf')
        closest_wp_ind = None
        closest_wp = None

        if self.base_waypoints is None:
            rospy.logwarn("WaypointUpdater: Trying to get closest before receiving base waypoints")
            return closest_wp_ind

        # loop through the waypoints and check for minimum distance
        for i, waypoint in enumerate(self.base_waypoints):
            sub_distance = self.__get_distance(waypoint.pose.pose, pose)
            if sub_distance < closest_distance:
                closest_wp_ind = i
                closest_wp = waypoint

        # get the closest waypoint ahead of the car
        if self.__is_behind(pose, closest_wp.pose.pose):
            if closest_wp_ind == len(self.base_waypoints)-1:
                return closest_wp_ind + 1

        return closest_wp_ind

    def __generate_waypoints(self, pose):
        '''
        To generate new waypoints from the car's current pose
        :param pose: the current pose of the car
        :return: a list of waypoints ahead of the car
        '''
        close_wp_ind = self.__get_closest_waypoint(pose)
        num_waypoints = len(self.base_waypoints)
        rospy.loginfo("Closest waypoint index is {}".format(close_wp_ind))

        # generate new waypoints by rotating the list
        next_waypoints = self.base_waypoints[close_wp_ind:] + self.base_waypoints[:(close_wp_ind + LOOKAHEAD_WPS) % num_waypoints]

        # TODO: Smoothen the velocity to the target velocity
        for waypoint in next_waypoints:
            waypoint.twist.twist.linear.x = 10.0

        return next_waypoints

    def __get_lane(self, frame_id, waypoints):
        '''
        To get the lane object for publishing
        :param frame_id: the current frame id
        :param waypoints: the new waypoints
        :return: the Lane object
        '''
        lane = Lane()
        lane.header.frame_id = frame_id
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints

        return lane

    def pose_cb(self, msg):
        '''
        Callback to handle pose updates. Generates the next waypoints
        :param msg: incoming message, contains pose data
        :return: None
        '''
        self.pose = msg.pose
        rospy.loginfo(
            "WaypointUpdater: Received current pose - Position is {},{},{} and orientation is {},{},{},{}".format(
                self.pose.position.x, self.pose.position.y, self.pose.position.z, self.pose.orientation.x,
                self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w))

        # check if base waypoints is loaded
        if self.base_waypoints is not None:
            # generate new waypoints
            waypoints = self.__generate_waypoints(self.pose)
            rospy.loginfo("Generated {} new waypoints".format(len(waypoints)))
            rospy.loginfo(
                "WaypointUpdater: Next waypoint - Position is {},{},{} and Vel is {}".format(
                    waypoints[0].pose.pose.position.x, waypoints[0].pose.pose.position.y, waypoints[0].pose.pose.position.z, waypoints[0].twist.twist.linear.x))
            # get the lane object
            lane = self.__get_lane(msg.header.frame_id, waypoints)
            # publish the waypoints
            self.final_waypoints_pub.publish(lane)
            rospy.loginfo("Published new waypoints")
        else:
            rospy.logwarn("WaypointUpdater: Received current pose before receiving base waypoints")

    def waypoints_cb(self, msg):
        '''
        Callback to handle incoming waypoints message.
        :param msg: incoming message, contains waypoints
        :return: None
        '''
        self.base_waypoints = msg.waypoints
        rospy.loginfo("WaypointUpdater: Received list of base waypoints. Total number of waypoints is {}".format(len(self.base_waypoints)))
        # unregister as we do not need this anymore
        self.base_wp_sub.unregister()

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

    def orthogonal_distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
