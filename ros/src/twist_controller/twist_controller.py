"""Controller class to handle throttle, brake and steering"""

__author__ = "Reza Arfa, Thomas Woodside"

import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    """
    Controller class to handle throttle, brake and steering
    """

    def __init__(self, pid_params, state_params):
        """

        :param throttle_pid_params: Params for the throttle PID
        :param brake_pid_params: Params for the brake PID
        :param state_params: the state params
        """

        # set the state parms
        self.brake_deadband = state_params["brake_deadband"]
        self.vehicle_mass = state_params["vehicle_mass"]
        self.wheel_radius = state_params["wheel_radius"]
        self.accel_limit = state_params["accel_limit"]
        self.decel_limit = state_params["decel_limit"]
        self.wheel_base = state_params["wheel_base"]
        self.steer_ratio = state_params["steer_ratio"]
        self.max_lat_accel = state_params["max_lat_accel"]
        self.max_steer_angle = state_params["max_steer_angle"]

        self.max_brake = self.vehicle_mass * abs(self.decel_limit) * self.wheel_radius

        self.prev_time = None

        # init controllers
        self.acceleration_controller = PID(pid_params["kp"], pid_params["ki"], pid_params["kd"])

        # init yaw controller.
        self.yaw_control = YawController(self.wheel_base, self.steer_ratio, 0., self.max_lat_accel,
                                         self.max_steer_angle)

    def control(self, t, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity,
                current_angular_velocity, dbw_enabled):
        """

        :param t: current time
        :param proposed_linear_velocity: desired linear velocity
        :param proposed_angular_velocity: desired angular velocity
        :param current_linear_velocity: current linear velocity
        :param current_angular_velocity: current angular velocity
        :param dbw_enabled: True if driving in autonomous mode
        :return: (throttle, brake, steering) tuple containing the target state
        """

        # if not in autonomous mode; reset the controllers
        if not dbw_enabled:
            self.prev_time = None
            self.acceleration_controller.reset()

        # if previous time is enabled
        if self.prev_time is not None:
            # calculate error in velocity and sample time
            delta_v = proposed_linear_velocity - current_linear_velocity
            delta_t = t - self.prev_time
	    self.prev_time = t

            throttle, brake = 0., 0.

	    torque = delta_v * self.vehicle_mass * self.wheel_radius
            acc = self.acceleration_controller.step(torque, delta_t)
	    if acc > 0.:
                throttle = min(acc, self.accel_limit)
            else:
		brake = min(abs(acc) + self.brake_deadband, self.max_brake)

	    
            # get the target steering
            steering = self.yaw_control.get_steering(proposed_linear_velocity, proposed_angular_velocity,
                                                     current_linear_velocity)

        else:
            # TODO: handle this case in a better way: maybe keep the current values?
            self.prev_time = t
            throttle, brake, steering = 0., 0., 0.

        # return the target values
        return throttle, brake, steering





