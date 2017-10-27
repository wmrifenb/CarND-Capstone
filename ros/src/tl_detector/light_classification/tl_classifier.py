import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
from model import Model

class TLClassifier(object):
    def __init__(self):
    
        # Counters for writing images
        self.red_image_number = 0
        self.yellow_image_number = 0
        self.green_image_number = 0
        self.unknown_image_number = 0
        

    def get_classification(self, image, traffic_light_state_truth):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # What... is the light
        if True:
            if traffic_light_state_truth == TrafficLight.RED:
                self.red_image_number += 1
                cv2.imwrite("new/red/image_" + str(self.red_image_number) + ".png", image)

            if traffic_light_state_truth == TrafficLight.YELLOW:
                self.yellow_image_number += 1
                cv2.imwrite("new/yellow/image_" + str(self.yellow_image_number) + ".png", image)
            
            if traffic_light_state_truth == TrafficLight.GREEN:
                self.green_image_number += 1
                cv2.imwrite("new/green/image_" + str(self.green_image_number) + ".png", image)
            
            if traffic_light_state_truth == TrafficLight.UNKNOWN:
                self.unknown_image_number += 1
                cv2.imwrite("new/none/image_" + str(self.unknown_image_number) + ".png", image)
        
        # Save
        return TrafficLight.UNKNOWN
        
