from styx_msgs.msg import TrafficLight

import tensorflow as tf
import cv2
import random
import rospy
import scipy.misc
import model_trainer
import numpy as np

class TLClassifier(object):
    def __init__(self):

        self.sess = None
        self.sess = tf.Session()

        training_mode = tf.placeholder(tf.bool)
        image_input_placeholder = tf.placeholder(tf.int8, (None, 128, 128, 3))
        image_input_layer = tf.image.convert_image_dtype(image_input_placeholder, tf.float32)

        # conv layers
        model_output = model_trainer.layers(image_input_layer, 3, training_mode)


        saver = tf.train.Saver()
        saver.restore(self.sess, "/home/student/VMDrive/CarND-Capstone/ros/src/tl_detector/light_classification/model.ckpt")

        for layer in [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]:
            rospy.loginfo(str(layer))

        self.image_tensor = image_input_placeholder
        self.training_mode = training_mode
        self.output_tensor = model_output
        pass

    count = 0
    def get_classification(self, image, light_state):

        #run classifier
        image_shape = (128, 128)
        image = scipy.misc.imresize(image, image_shape)

        results = self.sess.run([tf.nn.top_k(tf.nn.softmax(self.output_tensor))],
                           {self.training_mode: False, self.image_tensor: [image]})
        rospy.loginfo('rawresults:' + str(results))

        detected_light_state = int(np.array(results[0].indices).flatten()[0])

        rospy.loginfo('detected:' + str(detected_light_state) + " of " + str(type(detected_light_state)))
        if light_state != 4 and detected_light_state != light_state:
            image_id = random.randrange(0, 1000000)
            # save training image
            directory = "/home/student/VMDrive/CarND-Capstone/ros/src/tl_detector/trainer/train_data/t"+str(light_state)+"/image"
            image_name = directory + str(image_id) + ".jpg"
            cv2.imwrite(image_name, image)
            rospy.loginfo('savingImage incorrect:' + image_name)

        if detected_light_state == 0:
            return TrafficLight.RED
        if detected_light_state == 1:
            return TrafficLight.YELLOW
        if detected_light_state == 2:
            return TrafficLight.GREEN
        return TrafficLight.UNKNOWN


    def close(self):
        self.sess.close()