from styx_msgs.msg import TrafficLight
import cv2
import random
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image, light_state):
        if light_state != 4:
            image_id = random.randrange(0, 1000000)
            # save training image
            directory = "/home/student/VMDrive/images/t"+str(light_state)+"/image"
            image_name = directory + str(image_id) + ".jpg"
            cv2.imwrite(image_name, image)
            rospy.loginfo('savingImage:' + image_name)

        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
