
import rospy
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):

    	self.pid = PID(1, 1, 1)
    	self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    	self.previous_time = rospy.get_time()


    def control(self, linear_velocity, angular_velocity, current_linear_velocity, dbw_enabled):

    	# Run controllers
    	steering = 0
    	throttle = 0
    	if dbw_enabled:
	        # Get steering
	        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_linear_velocity)

	        # Get throttle
	    	throttle = self.pid.step(linear_velocity - current_linear_velocity, rospy.get_time() - self.previous_time)
        self.previous_time = rospy.get_time()

        # Calculate braking torque
        brake = 0
        if throttle < 0:
        	brake = -throttle
        	throttle = 0

        # Return throttle, brake, steering
        return throttle, brake, steering
