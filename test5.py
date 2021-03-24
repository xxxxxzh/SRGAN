import numpy as np
import cv2
from keras import backend as K
from keras import models, layers, optimizers, losses
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
import os

from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
import time

patch_size = 96


# Residual block
def residual_block(input_model, kenal_size, filter_num, strides):
    x = Conv2D(filter_num, kenal_size, strides=strides, padding='same')(input_model)
    x = BatchNormalization(momentum=0.5)(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(x)
    x = Conv2D(filter_num, kenal_size, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = add([input_model, x])
    return x


def generator_upsample(input_model, kenal_size, filter_num, strides):
    x = input_model
    x = Conv2D(filter_num, kenal_size, strides=strides, padding='same')(x)
    x = UpSampling2D(size=2)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(x)
    return x


def discriminator_block(input_model, kenal_size, filter_num, strides):
    x = input_model
    x = Conv2D(filter_num, kenal_size, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def get_generator(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), strides=1, padding='same')(input)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(x)
    model = x
    for i in range(16):
        x = residual_block(x, (3, 3), 64, 1)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = add([model, x])
    for i in range(2):
        x = generator_upsample(x, (3, 3), 256, 1)
    x = Conv2D(3, (9, 9), strides=1, padding='same')(x)
    output = Activation('tanh')(x)
    generator_model = Model(input, output)
    return generator_model


def get_discriminator(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = discriminator_block(x, 3, 64, 2)
    x = discriminator_block(x, 3, 128, 1)
    x = discriminator_block(x, 3, 128, 2)
    x = discriminator_block(x, 3, 256, 1)
    x = discriminator_block(x, 3, 256, 2)
    x = discriminator_block(x, 3, 512, 1)
    x = discriminator_block(x, 3, 512, 2)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)
    discriminator_model = Model(input, output)
    return discriminator_model


def get_optimizer():
    return Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


class VGGLOSS(object):
    def __init__(self, img_shape):
        self.shape = img_shape

    def vggloss(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(y_true) - model(y_pred)))

def img_normalize(img):
    return (img.astype('float32') - 127.5) / 127.5

def img_denormalize(img):
    return ((img + 1) * 127.5).astype(np.uint8)

def img_norm(img):
    #img = img * 255
    img[img[:] > 255] = 255
    img[img[:] < 0] = 0
    img = img.astype('uint8')
    return img

def PSNRLOSS(y_true, y_pred):
    return 48.1308036087 - 10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)
    #return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.) #归一化后的计算公式

def predict(generator_path,input_path,output_path,count=1):
    generator = get_generator((None,None,3))
    adam = get_optimizer()
    loss = VGGLOSS((None,None,3))
    generator.compile(loss=loss.vggloss, optimizer=adam)
    generator.load_weights(generator_path)
    true_img = image.img_to_array(image.load_img(input_path,target_size=(384,384)))
    input_img = image.img_to_array(image.load_img(input_path,target_size=(96,96)))
    input_img = img_normalize(input_img)
    input_img = input_img.reshape((1,) + input_img.shape)
    output_img = generator.predict(input_img)
    shape = output_img.shape
    output_img = output_img.reshape((shape[1],shape[2],shape[3]))
    output_img = img_denormalize(output_img)

    print('PSNR: %f\n' %PSNRLOSS(true_img,output_img))

    output_img = img_norm(output_img)
    output_img = image.array_to_img(output_img)

    shape = input_img.shape
    input_img = input_img.reshape((shape[1],shape[2],shape[3]))
    input_img = img_denormalize(input_img)
    input_img = img_norm(input_img)
    input_img = image.array_to_img(input_img)

    true_img = image.array_to_img(true_img)
    image.save_img(os.path.join(output_path,'input_{}.jpg'.format(count)), input_img)
    image.save_img(os.path.join(output_path, 'pred_{}.jpg'.format(count)), output_img)
    image.save_img(os.path.join(output_path, 'true_{}.jpg'.format(count)), true_img)

output_dir = 'E:\\anaconda3\\envs\\tensorflow\\datasets\\SRCNN_Trainning\\SRGAN_result'
predict_dir = 'E:\\anaconda3\\envs\\tensorflow\\datasets\\SRCNN_Trainning\\Test\\set14\\baboon.bmp'

predict('Models/generator_epoch_1500.h5',predict_dir,output_dir,13)