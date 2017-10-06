import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import cv2
import time
import tensorflow as tf
from glob import glob
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
from PIL import ImageEnhance, Image, ImageOps


def distort_image(image):

    adjust_c = (random.random() * 0.2) + 0.9
    adjust_b = (random.random() * 0.2) + 0.9
    c_enhance = ImageEnhance.Contrast(Image.fromarray(image)).enhance(adjust_c)
    b_enhance = ImageEnhance.Brightness(c_enhance).enhance(adjust_b)
    if random.random() > 0.5:
        final = ImageOps.mirror(b_enhance)
    else:
        final = b_enhance
    #add some skew here
    image_array = np.array(final)
    rows, cols, ch = image_array.shape

    pts1 = np.float32([[5, 5], [95, 5], [5, 95]])
    pts2 = np.float32([[5 + random.randint(-3,3), 5+ random.randint(-3,3)],
                       [95 + random.randint(-3,3), 5+ random.randint(-3,3)],
                       [5+ random.randint(-3,3), 95+ random.randint(-3,3)]])

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(image_array, M, (cols, rows))
    return dst


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = glob(os.path.join(data_folder, 't*', '*.jpg'))
    for image_file in image_paths:
        original = scipy.misc.imread(image_file)
        image = scipy.misc.imresize(original, image_shape)
        if original.shape != image.shape:
            scipy.misc.imsave(image_file, image)

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            labels = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:

                image = scipy.misc.imread(image_file)
                label = int(os.path.dirname(image_file)[-1])

                #augment the image
                image = distort_image(image)
                images.append(image)
                labels.append(label)

            images_np = np.array(images)
            labels_np = np.array(labels)
            #print("images_size:"+str(images_np.shape))
            #print("labels_size:" + str(labels_np.shape))
            yield images_np, labels_np

    return get_batches_fn

def run_test_data(data_folder, image_shape, input_image, logits, count, training_mode, sess):

    image_paths = glob(os.path.join(data_folder, 't*', '*.jpg'))
    random.shuffle(image_paths)
    correct = 0
    chunk_size = 5
    for batch_i in range(0, count, chunk_size):
        images = []
        labels = []
        for image_file in image_paths[batch_i:batch_i + chunk_size]:
            original = scipy.misc.imread(image_file)
            image = scipy.misc.imresize(original, image_shape)
            if original.shape != image.shape:
                scipy.misc.imsave(image_file, image)
            label = int(os.path.dirname(image_file)[-1])
            images.append(np.array(image))
            labels.append(label)
        images_np = np.array(images)
        labels_np = np.array(labels)
        #print("images_size:" + str(images_np.shape))
        #print("labels_size:" + str(labels_np.shape))

        results = sess.run([tf.nn.top_k(tf.nn.softmax(logits))],
            {input_image: images_np, training_mode: False})
        indices = np.array(results[0].indices).flatten()

        for idx in range(chunk_size):
            index = indices[idx]
            label = labels_np[idx]
            if index == label:
                correct += 1
            print(str(idx) + ":" + str(index) + " " + str(label))
    print("accuracy:" + str(correct/(count*1.0)))

    return indices, labels_np