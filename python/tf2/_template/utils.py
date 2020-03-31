import os
import json
from bunch import Bunch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def data_loader(config):

    def normalize(data):
        image = tf.cast(data['image'], tf.float32)
        image = image / 255.
        return image

    data = tfds.load("mnist")
    train_data = data['train']
    trainloader = train_data.map(normalize).shuffle(60000).batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return trainloader

def save_images(images, path):
    num_samples, h, w, c = images.shape[0], images.shape[1], images.shape[2], images.shape[3]
    frame_dim = int(np.sqrt(num_samples))
    canvas = np.squeeze(np.zeros((h * frame_dim, w * frame_dim, c)))
    for idx, image in enumerate(images):
        i = idx // frame_dim
        j = idx % frame_dim
        if c==1:
            canvas[i*h : (i+1)*h, j*w : (j+1)*w] = np.squeeze(image)
        elif c==3:
            canvas[i*h : (i+1)*h, j*w : (j+1)*w, :] = image
        else:
            print('Image channels must be 1 or 3!')
    if c==1:
        plt.imsave(path, canvas, cmap='gray')
    if c==3:
        plt.imsave(path, canvas)

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
