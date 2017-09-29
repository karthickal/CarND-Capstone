#!/usr/bin/env python
import rospy
import sys
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import numpy as np
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub4 = rospy.Subscriber('/final_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

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
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        best = sys.float_info.max
        closest = 0

        if not self.waypoints:
            return 0
        for idx, wp in enumerate(self.waypoints.waypoints):
            waypoint_pose = wp.pose.pose.position
            dist = self.distance(pose, waypoint_pose)
            if dist < best:
                closest = idx
                best = dist
            else:  # If we are getting farther away, stop. In this track we are generally either approaching or receding.
                break

        return closest, best

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def distance(self, pose1, pose2):
        xcomp = math.pow(pose2.x - pose1.x, 2)
        ycomp = math.pow(pose2.y - pose1.y, 2)
        zcomp = math.pow(pose2.z - pose1.z, 2)
        dist = math.sqrt(xcomp + ycomp + zcomp)
        return dist

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

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose and self.waypoints and self.waypoints.waypoints:
            car_position, __ = self.get_closest_waypoint(self.pose.pose.position)
            rospy.loginfo('closest car waypoint:' + str(car_position))

            #TODO find the closest visible traffic light (if one exists)
            closest = 200
            for stop_line in stop_line_positions:
                pos = Point()
                pos.x = stop_line[0]
                pos.y = stop_line[1]
                pos.z = self.pose.pose.position.z

                # prevent expensive closest point calc when not needed.
                if self.distance(pos, self.pose.pose.position) > 200:
                    continue

                wp, dist = self.get_closest_waypoint(pos)
                rospy.loginfo('closest waypoint:' + str(wp) + " dist:" + str(dist) + " for:" + str(stop_line))

                if dist < 1 and wp < 150 and wp < closest:
                    closest = wp
                    line_waypoint = self.waypoints.waypoints[wp].pose.pose.position
                    line_waypoint_next = self.waypoints.waypoints[wp+2].pose.pose.position
                    best = sys.float_info.max
                    #TODO: only use it if the light is infront of me.
                    for possible_light in self.lights:
                        dist_light = self.distance(possible_light.pose.pose.position, line_waypoint)
                        rospy.loginfo('dist_light:' + str(dist_light))
                        if dist_light < 50 and dist_light < best:
                            dist_light_next = self.distance(possible_light.pose.pose.position, line_waypoint_next)
                            rospy.loginfo('dist_next_light' + str(dist_light_next))

                            if dist_light_next < dist_light:
                                best = dist_light
                                light = possible_light
                                light_wp = wp
                                rospy.loginfo('using_light')
                            # find light closest to stop line (may need to project forward)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')