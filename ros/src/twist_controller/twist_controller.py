import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, kp, ki, kd, max_lat_accel, max_steer_angle, steer_ratio, wheel_base):
        self.brake_deadband = rospy.get_param('~brake_deadband',  0.2)
        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)

        self.prev_time = None

        # init acceleration controlers
        self.acceleration_controller = PID(kp, ki, kd)  # PID for acceleration

        # init yaw controller.
        self.yaw_control = YawController(wheel_base, steer_ratio, 0., max_lat_accel, max_steer_angle)

        # init low-pass filters (lpf) to filter high frequency signals and smooth
        # TODO: need to set these values
        self.throttle_lpf = LowPassFilter(0.5, 0.5) # smoother throttle command
        self.brake_lpf = LowPassFilter(0.5, 0.5) # smoother brake command
        self.steering_lpf = LowPassFilter(1.0, 1.0) # smoother steering command

    def control(self, t, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity,
                current_angular_velocity, dbw_enabled):
        '''
        INPUTS
            t 			  : current time
            proposed_linear_velocity  : desired linear velocity
            proposed_angular_velocity : desired angular velocity
            current_linear_velocity   : current linear velocity
            current_angular_velocity  : current angular velocity
            dbw_enabled  	          : True if driving in autonomous mode
        OUPUTS
            throttle	: proposed throttle value
            brake		: proposed brake value
            steering	: proposed steering value
        '''
        if not dbw_enabled:
            self.prev_time = None
            self.acceleration_controller.reset()
           
        if self.prev_time is not None:
            delta_t = self.get_dt(t)

            # get throttle and brake
            delta_v = proposed_linear_velocity - current_linear_velocity
            acceleration = self.acceleration_controller.step(delta_v, delta_t)
            throttle, brake = self.acceleration_to_brake_throttle(acceleration)

            # get steering
            steering = self.yaw_control.get_steering(proposed_linear_velocity, proposed_angular_velocity,
                                                     current_linear_velocity)

            # TODO: smooth the signals if needed
            throttle = self.throttle_lpf.filt(throttle)
            brake = self.brake_lpf.filt(brake)
            steering = self.steering_lpf.filt(steering)

        else:
            # TODO: handle this case in a better way: maybe keep the current values?
            self.prev_time = t
            throttle, brake, steering = 0., 0., 0.

        return throttle, brake, steering

    def get_dt(self, t):
        delta_t = t - self.prev_time
        self.prev_time = t
        return delta_t

    def acceleration_to_brake_throttle(self, acceleration):
        '''translates acceleration value to throttle and brake values'''
        throttle, brake = 0., 0.

        if acceleration >= 0.:
            throttle = acceleration
        else:
            brake = acceleration * self.vehicle_mass * self.wheel_radius + self.brake_deadband
        return throttle, brake

