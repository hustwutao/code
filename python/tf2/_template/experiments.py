import os
import numpy as np
import tensorflow as tf
from utils import save_images, make_dirs

class Experiments(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

        # directory to save experimental results
        self.dir = make_dirs(os.path.join(self.config.result_path, self.config.experiment_path))

    def image_generation(self):

        # the number of samples
        nx = ny = 10
        num_samples = nx * ny

        # sampling z from N(0, 1)
        z_samples = np.random.normal(0, 1, (num_samples, self.config.z_dim))

        # generate images
        samples = self.model.gen(z_samples)

        # save images
        path = make_dirs(os.path.join(self.dir, 'image_generation')) + '/mnist.png'
        save_images(samples, path)
