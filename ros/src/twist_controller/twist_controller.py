
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        pass

    def control(self, linear_velocity, angular_velocity, current_linear_velocity, dbw_enabled):

        # Return throttle, brake, steering
        return linear_velocity/20, 0., angular_velocity
