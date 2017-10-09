#!/usr/bin/env python
import rospy
import sys
from std_msgs.msg import Int32, Header
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
import scipy.misc
from tf.transformations import euler_from_quaternion

STATE_COUNT_THRESHOLD = 3
VISIBLE_DISTANCE = 85.0

class TLDetector(object):
    def __init__(self):
        self.started = False
        rospy.init_node('tl_detector')

        self.pose = None
        self.base_waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, buff_size=1000000, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, buff_size=8000000, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # register publisher to broadcast traffic waypoint
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.listener = tf.TransformListener()

        # set defaults and params
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.traffic_map = {}

        self.light_classifier = TLClassifier()
        self.started = True
        rospy.spin()

    def distance(self, pose1, pose2):
        """
        To calculate the distance between two poses
        :param pose1 (Point): the origin
        :param pose2 (Point): the target
        :return: distance between the poses
        """
        xcomp = math.pow(pose2.x - pose1.x, 2)
        ycomp = math.pow(pose2.y - pose1.y, 2)
        zcomp = math.pow(pose2.z - pose1.z, 2)
        dist = math.sqrt(xcomp + ycomp + zcomp)
        return dist

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

        if not self.base_waypoints:
            return -1
        for idx, wp in enumerate(self.base_waypoints.waypoints):
            waypoint_position = wp.pose.pose.position
            dist = self.distance(pose.position, waypoint_position)
            if dist < best:
                closest = idx
                best = dist

        return closest, best

    def load_traffic_map(self):
        """
        To load the nearest waypoints to the traffic line positions. Will be triggered as soon as base waypoints are set.
        :return: None
        """
        # check if base waypoints are set
        if self.base_waypoints is None:
            rospy.logwarn("TLDetector: Trying to load nearest waypoints before receiving base waypoints/light positions")
            return None

        # for each stop line position get the nearest waypoint and store in a dict
        for light in self.config['stop_line_positions']:
            pose = Pose()
            pose.position.x = light[0]
            pose.position.y = light[1]
            pose.position.z = 0.0
            closest_waypoint, _ = self.get_closest_waypoint(pose)
            self.traffic_map[closest_waypoint] = pose

    def is_behind(self, pose, target_wp):
        """
        To determine if the target waypoint is behind the car
        :param pose: the pose of the car
        :param target_wp: the target waypoint
        :return: bool True if waypoint is behind else False
        """

        # get the yaw angle after transformation
        _, _, yaw = euler_from_quaternion([pose.orientation.x,
                                           pose.orientation.y,
                                           pose.orientation.z,
                                           pose.orientation.w])

        origin_x = pose.position.x
        origin_y = pose.position.y

        # shift the co-ordinates
        shift_x = target_wp.pose.pose.position.x - origin_x
        shift_y = target_wp.pose.pose.position.y - origin_y

        # rotate and check orientation
        x = (shift_x * math.cos(0.0 - yaw)) - (shift_y * math.sin(0.0 - yaw))
        if x > 0.0:
            return False
        return True


    def get_closest_traffic_light(self, origin_idx):
        """
        Method to get the closest traffic light waypoint
        :param origin_idx: the waypoint closest to the car
        :return: the closest traffic waypoint, its index
        """

        # get the origin waypoint and traffic map
        origin_wp = self.base_waypoints.waypoints[origin_idx]
        traffic_map = dict(self.traffic_map)

        # check the traffic signal closest and ahead of the origin
        best_distance = float('inf')
        best_tl = None
        best_wp_idx = None
        for target_idx, pose in traffic_map.iteritems():
            distance = self.distance(origin_wp.pose.pose.position, pose.position)
            if distance < best_distance:
                # if waypoint is behind, skip
                if self.is_behind(origin_wp.pose.pose, self.base_waypoints.waypoints[target_idx]):
                    continue
                best_distance = distance
                best_tl = pose
                best_wp_idx = target_idx

        # check if traffic light is visible
        if best_distance > VISIBLE_DISTANCE:
            return None, -1

        return best_tl, best_wp_idx

    def get_closest_light(self, light_pose):
        """
        Get closest reported light to target position.
        :param light_pose: approx position of the target traffic light
        :return: closest traffic light
        """
        best_dist = float('inf')
        closest_light = None
        for light in self.lights:
            distance = self.distance(light_pose.position, light.pose.pose.position)
            if distance < best_dist:
                best_dist = distance
                closest_light = light

        return closest_light

    def pose_cb(self, msg):
        """
        Callback to handle incoming pose messages. Message consists of the current pose of the car
        :param msg: the incoming message
        :return: None
        """
        self.pose = msg
        self.update_lights()

    def base_waypoints_cb(self, waypoints):
        """
        Callback to handle incoming map waypoint message. Message contains a complete list of waypoints on the map.
        :param waypoints: the incoming message
        :return: None
        """
        self.base_waypoints = waypoints
        self.load_traffic_map()

    def traffic_cb(self, msg):
        """
        Callback to handle incoming traffic light messages.
        :param msg: the incoming message
        :return: None
        """
        self.lights = msg.lights

    image_processed = False
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        if not self.started:
            return

        self.has_image = True
        self.camera_image = msg

        self.image_processed = False
        self.update_lights()

    def update_lights(self):
        if not self.pose or not self.camera_image:
            rospy.logwarn('state missing for light update')
            return

        if self.image_processed:
            rospy.logdebug('image already processed')
            return

        if (self.pose.header.stamp - self.camera_image.header.stamp).nsecs > 200000000:
            rospy.loginfo("skipping light update - image and position not in synch")
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            return

        rospy.loginfo("Updating traffic light")
        light_wp, state = self.process_traffic_lights()
        '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''

        self.image_processed = True

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

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world
        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image
        """
        #General formula and offsets were used from discussion on https://discussions.udacity.com/t/focal-length-wrong/358568/23

        # retrieve config values
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
        image_center_x_offset = image_width / 2
        image_center_y_offset = image_height / 2

        ##########################################################################################
        # overrides for simulator camera
        if fx < 10:
            fx = 2574
            fy = 2744
            image_center_x_offset = (image_width / 2) - 30
            image_center_y_offset = image_height + 50
        ##########################################################################################

        # get transform between pose of camera and world frame
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                                           "/world", now, rospy.Duration(1.0))
            (transT, rotT) = self.listener.lookupTransform("/base_link",
                                                           "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")
            return None, None, None, None, None, None

        # get car orientation - yaw
        euler_vector = tf.transformations.euler_from_quaternion(rotT)
        yaw = euler_vector[2]

        # Rotate the coordinate space on the z axis to align relative to car
        world_x = point_in_world.x
        world_y = point_in_world.y
        world_z = point_in_world.z

        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        yaw_oriented_point = (world_x * cos_yaw - world_y * sin_yaw,
                              world_x * sin_yaw + world_y * cos_yaw,
                              world_z)

        # Apply transformation to center on car position
        car_rel_x = yaw_oriented_point[0] + transT[0]
        car_rel_y = yaw_oriented_point[1] + transT[1]
        car_rel_z = yaw_oriented_point[2] + transT[2]

        # rotate to camera view space with offset
        camera_height_offset = 1.1
        camera_rel_x = -car_rel_y
        camera_rel_z = car_rel_x
        camera_rel_y = -(car_rel_z - camera_height_offset)

        # apply focal length and offsets
        #  center
        center_x = int((camera_rel_x * fx / camera_rel_z) + image_center_x_offset)
        center_y = int((camera_rel_y * fy / camera_rel_z) + image_center_y_offset)

        corner_y_offset = 1.5
        corner_x_offset = 1.5

        #  top left
        top_x = int(((camera_rel_x - corner_x_offset) * fx / camera_rel_z) + image_center_x_offset)
        top_y = int(((camera_rel_y - corner_y_offset) * fy / camera_rel_z) + image_center_y_offset)

        #  bottom right
        bottom_x = int(((camera_rel_x + corner_x_offset) * fx / camera_rel_z) + image_center_x_offset)
        bottom_y = int(((camera_rel_y + corner_y_offset) * fy / camera_rel_z) + image_center_y_offset)

        return [top_x, top_y, bottom_x, bottom_y, center_x, center_y]

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # mycoords
        [top_x, top_y, bottom_x, bottom_y, center_x, center_y] = self.project_to_image_plane(light.pose.pose.position)
        if top_x < 0 or top_y < 0 or bottom_y >= 600 or bottom_x >= 800:
            return False

        # preprocess the image
        croppedImage = cv_image[top_y:bottom_y, top_x:bottom_x]
        classifier_shape = (128, 128)
        final_image = scipy.misc.imresize(croppedImage, classifier_shape)

        # Get classification of the pre-processed image
        return self.light_classifier.get_classification(final_image, light.state)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        car_wp_idx, _ = self.get_closest_waypoint(self.pose.pose)
        traffic_pose, traffic_idx = self.get_closest_traffic_light(car_wp_idx)
        if traffic_idx != -1:
            rospy.logdebug("TLDetector: Traffic Light at {}".format(traffic_idx))
            light = self.get_closest_light(traffic_pose)
            state = self.get_light_state(light)

            return traffic_idx, state

        rospy.logdebug("TLDetector: Traffic Light not found")
        return -1, TrafficLight.UNKNOWN

    def create_pose(self, x, y, z, yaw=0.0):
        """
        Create a quarternion pose object from euler co-ordinates
        :param x: position x
        :param y: position y
        :param z: position z
        :param yaw: yaw angle
        :return: the pose object
        """
        pose = PoseStamped()

        pose.header = Header()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'world'

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        return pose


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
