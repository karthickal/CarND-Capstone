"""Controller class to handle the yaw angle"""

__author__ = "Reza Arza, Thomas Woodside"

from math import atan

class YawController(object):
    """
    Controller class to handle the yaw angle
    """
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        """
        Initialize the controller
        :param wheel_base: the wheel base of the vehicle
        :param steer_ratio: the steering ration
        :param min_speed: the minimum speed
        :param max_lat_accel: the maximum lateral acceleration
        :param max_steer_angle: the maximum steering angle
        """

        # initialize properties
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        # calculate the minimum and maximum angle
        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle


    def get_angle(self, radius):
        """
        Calculate the angle from the radius
        :param radius: input radius
        :return: angle
        """
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_velocity, angular_velocity, current_velocity):
        """
        Method to get the target steering
        :param linear_velocity: the desired linear velocity
        :param angular_velocity: the desired angular velocity
        :param current_velocity: the current velocity
        :return:
        """

        # get the target angular velocity if desired velocity is greater than 0
        angular_velocity = current_velocity * angular_velocity / linear_velocity if abs(linear_velocity) > 0. else 0.
        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity)
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))

        # return the steering angle
        return self.get_angle(max(current_velocity, self.min_speed) / angular_velocity) if abs(angular_velocity) > 0. else 0.0;
