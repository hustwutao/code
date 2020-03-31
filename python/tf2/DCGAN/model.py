import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.gen = Sequential([
            layers.InputLayer(input_shape=(self.config.z_dim,)),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(7 * 7 * 128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='sigmoid'),
            ])

    def call(self, z):
        x = self.gen(z)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.dis = Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1),
            ])

    def call(self, x):
        z = self.dis(x)
        return z


class DCGAN(object):
    def __init__(self, config):
        self.config = config
        self.dis = Discriminator(self.config)
        self.gen = Generator(self.config)
        self.d_optim = tf.keras.optimizers.Adam(self.config.learning_rate, 0.5)
        self.g_optim = tf.keras.optimizers.Adam(self.config.learning_rate, 0.5)

    def loss(self, x_train):

        z = tf.random.normal([self.config.batch_size, self.config.z_dim])
        g_fake = self.gen(z)
        d_fake = self.dis(g_fake)
        d_real = self.dis(x_train)

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        d_loss_real = cross_entropy(tf.ones_like(d_real), d_real)
        d_loss_fake = cross_entropy(tf.zeros_like(d_fake), d_fake)
        d_loss = d_loss_real + d_loss_fake
        g_loss = cross_entropy(tf.ones_like(d_fake), d_fake)

        return d_loss, g_loss