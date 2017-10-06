import os.path
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
import time
import numpy as np

import helper
import warnings
from distutils.version import LooseVersion


# Check TensorFlow Version
from tensorflow.python.saved_model import tag_constants

assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def convLayer(inputLayer, depth, width, stride):

    #first layer
    decoder_1 = tf.layers.conv2d(inputLayer, depth, width, strides=1, padding='same',
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool_1 = tf.layers.max_pooling2d(decoder_1, stride, stride, padding='same')
    return pool_1

def layers(image_input, numclasses, training_mode):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    normalized = tf.layers.batch_normalization(image_input)

    #1
    decoder_1 = convLayer(normalized, 128, 5, 2)
    dropout_1 = tf.layers.dropout(decoder_1, rate=0.75, training=training_mode)

    #2
    decoder_2 = convLayer(dropout_1, 64, 5, 2)
    dropout_2 = tf.layers.dropout(decoder_2, rate=0.75, training=training_mode)

    #3
    decoder_3 = convLayer(dropout_2, 32, 5, 2)

    #4
    decoder_4 = convLayer(decoder_3, 32, 5, 2)

    #5
    decoder_5 = convLayer(decoder_4, 32, 5, 2)

    #6
    decoder_6 = convLayer(decoder_5, 32, 5, 2)

    #7
    decoder_7 = convLayer(decoder_6, 32, 5, 2)
    output = tf.layers.conv2d(decoder_7, numclasses, 1, strides=(1, 1), padding='valid',
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output


def optimize(logits, labels, learning_rate):
    """
    Build the TensorFLow loss and optimizer operations.
    :param logits: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, loss)
    """


    sess = tf.Session()
    op = sess.graph.get_operations()

    for layer in [m.values() for m in op]:
        print(layer)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="softmax_entropy"))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss)

    return logits, train_op, loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, learning_rate, training_mode, saver):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """


    # load saved weights
    if os.path.isfile("./model.ckpt.index"):
        saver.restore(sess, "./model.ckpt")
        print("Model restored.")
    else:
        sess.run(tf.global_variables_initializer())

    count = 0
    for current_epoch in range(epochs):
        timestamp = time.time()

        num_samples = 0
        epoch_loss = 0
        count = (count + 1) % 10

        for batch_x, batch_y in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
               feed_dict={input_image: batch_x, learning_rate: 0.0001, correct_label: batch_y, training_mode: True})
            #print("batch loss: " + str(loss) + " count: " + str(len(batch_x)))

            num_samples += len(batch_x)
            epoch_loss += loss

        total_time = time.time() - timestamp
        epoch_avg_loss = epoch_loss / num_samples

        print("num_samples: " + str(num_samples))
        print("current_epoch: {}".format(current_epoch))
        print("time: %.2f sec" % total_time)
        print("avg. loss: {:.7f}".format(epoch_avg_loss))
        print()

        if count == 9:
            save_path = saver.save(sess, "./model.ckpt")
            print("Model saved in file: %s" % save_path)
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)

def run():
    num_classes = 3
    image_shape = (128, 128)
    epochs = 200
    batch_size = 100
    data_dir = './data'
    test_data_dir = './fulldata'

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        #placeholders
        learning_rate_placeholder = tf.placeholder(tf.float32)
        training_mode = tf.placeholder(tf.bool)
        image_input_placeholder = tf.placeholder(tf.int8, (None, 128, 128, 3))
        image_input_layer = tf.image.convert_image_dtype(image_input_placeholder, tf.float32)
        label_placeholder = tf.placeholder(tf.int32, None)

        # batches
        get_batches_fn = helper.gen_batch_function(data_dir, image_shape)

        # conv layers
        model_output = layers(image_input_layer, num_classes, training_mode)

        # get optimizer
        logits, training_op, cross_entropy_loss = optimize(model_output, label_placeholder, learning_rate_placeholder)

        saver = tf.train.Saver()

        # train on generated data
        train_nn(sess, epochs, batch_size, get_batches_fn, training_op, cross_entropy_loss, image_input_placeholder, label_placeholder, learning_rate_placeholder, training_mode, saver)

        helper.run_test_data(test_data_dir, image_shape, image_input_placeholder, model_output, 100, training_mode, sess)


def _build_classification_signature(input_tensor, scores_tensor):
  """Helper function for building a classification SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {
      tf.saved_model.signature_constants.CLASSIFY_INPUTS: input_tensor_info
  }
  output_tensor_info = tf.saved_model.utils.build_tensor_info(scores_tensor)
  signature_outputs = {
      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
          output_tensor_info
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

if __name__ == '__main__':
    run()
