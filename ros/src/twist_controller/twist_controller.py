import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
from std_msgs.msg import Float32

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, max_lat_accel, max_steer_angle, steer_ratio, wheel_base):
        self.brake_deadband = rospy.get_param('~brake_deadband',  0.2)

        self.prev_time = None

	# subscribe to Kp, Ki, and Kd
        rospy.Subscriber('/kp', Float32, self.kp_callback) # P
        rospy.Subscriber('/ki', Float32, self.ki_callback) # I
        rospy.Subscriber('/kd', Float32, self.kd_callback) # D

	# init controlers
	self.pid_acceleration  = PID(3.0, 0.5, 0.01) # PID for acceleration

	# init low-pass filters (lpf) to filter high frequency signals and smooth 
        self.lpf = LowPassFilter(0.5, 0.5)

	# init YawController to obtain steering yaw. 
        self.yaw_control  = YawController(wheel_base=wheel_base, 
                                          steer_ratio=steer_ratio,
                                          min_speed=0., 
                                          max_lat_accel=max_lat_accel,
                                          max_steer_angle=max_steer_angle)    


    def control(self, t, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, current_angular_velocity, dbw_enabled):
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
	# TODO: 

        if not dbw_enabled:
            self.pid_acceleration.reset()
            self.pid_steering.reset()
           
        if self.prev_time is not None:
   	    delta_t = self.get_dt(t)

	    # get throttle and brake
            delta_v = proposed_linear_velocity - current_linear_velocity
            acceleration = self.pid_acceleration.update(delta_v, delta_t)
	    throttle, brake = self.acceleration_to_brake_throttle(acceleration)       

	    # get steering
            raw_steering = self.yaw_control.get_steering(proposed_linear_velocity, proposed_angular_velocity,  current_linear_velocity)
            steering = self.lpf.filter(raw_steering)

        else:
            self.prev_time = t
	    throttle, brake, steering = 0., 0., 0.

        return throttle, brake, steering

    def get_dt(self, t):
	delta_t = t - self.prev_time
        self.prev_time = t 
	return delta_t

    def acceleration_to_brake_throttle(self, acceleration):
	'''translates acceleration value to throttle and brake values'''
	deceleartion  = -acceleration
	throttle = max(0., acceleration)
        brake = max(0., deceleartion) + self.brake_deadband
	return throttle, brake 

    def kp_callback(self, msg):
        self.pid_steering.kp = msg.data

    def ki_callback(self, msg):
        self.pid_steering.ki = msg.data

    def kd_callback(self, msg):
        self.pid_steering.kd = msg.data

