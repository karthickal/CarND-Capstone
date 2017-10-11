#!/usr/bin/env python

""" The DBW node"""

__author__ = "Reza Arfa, Thomas Woodside"

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''


class DBWNode(object):
    def __init__(self):
        '''
        Initialize the DBW node
        '''
        rospy.init_node('dbw_node')

        # get states
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.twist_cmd = None
        self.current_velocity = None
        self.dbw_enabled = None

        # publish steering command, throttle command, and brake command
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',  BrakeCmd, queue_size=1)

        # parameters for throttle controller
        pid_params = {}
        pid_params["kp"] = 5.0
        pid_params["ki"] = 0.0
        pid_params["kd"] = 0.001

        # state parameters for controller
        state_params = {}
        state_params["brake_deadband"] = brake_deadband
        state_params["vehicle_mass"] = vehicle_mass
        state_params["wheel_radius"] = wheel_radius
        state_params["accel_limit"] = accel_limit
        state_params["decel_limit"] = decel_limit
        state_params["wheel_base"] = wheel_base
        state_params["steer_ratio"] = steer_ratio
        state_params["max_lat_accel"] = max_lat_accel
        state_params["max_steer_angle"] = max_steer_angle

        # Create `TwistController` object
        self.controller = Controller(pid_params, state_params)

        # subscribe to `DBW status`->Bool, `current velocity`->TwistStamped, and `twist command`->TwistStamped
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_callback)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_callback)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_callback)

        # run the loop
        self.loop()

    def loop(self):
        """
        Publish the target states
        :return: None
        """
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            if (not self.twist_cmd) or (not self.current_velocity):  # first commands are not published yet.
                continue

            t = rospy.get_time()  # current time

            # get the target velocities
            proposed_linear_velocity = self.twist_cmd.twist.linear.x
            proposed_angular_velocity = self.twist_cmd.twist.angular.z

            # get the current velocities
            current_linear_velocity = self.current_velocity.twist.linear.x
            current_angular_velocity = self.current_velocity.twist.angular.z

            # get the desired next state
            throttle, brake, steering = self.controller.control(t,
                                                                proposed_linear_velocity,
                                                                proposed_angular_velocity,
                                                                current_linear_velocity,
                                                                current_angular_velocity,
                                                                self.dbw_enabled    	  # reset controller if false
                                                                )
            # publish state is DBW is enabled
            if self.dbw_enabled:
                self.publish(throttle, brake, steering)

            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def current_velocity_callback(self, msg):
        """
        Callback to handle incoming velocity message
        :param msg: the velocity
        :return: None
        """
        self.current_velocity = msg

    def twist_cmd_callback(self, msg):
        """
        Callback to handle the twist messages
        :param msg: the incoming twist message
        :return: None
        """
        self.twist_cmd = msg

    def dbw_enabled_callback(self, msg):
        """
        Callback to handle DBW state
        :param msg: incoming message containing DBW state
        :return: None
        """
        self.dbw_enabled = msg.data


if __name__ == '__main__':
    DBWNode()
