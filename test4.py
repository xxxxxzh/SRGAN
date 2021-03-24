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



def get_gan(discriminator, generator, shape, optmizier, loss):
    # discriminator.trainable = False
    input = Input(shape=shape)
    x = generator(input)

    #只取一块32的图像块
    y = make_array_patchs(x)
    output = discriminator(y)

    #output = discriminator(x)
    gan_model = Model(input, [x, output])
    gan_model.compile(loss=[loss, 'binary_crossentropy'], loss_weights=[1., 1e-3], optimizer=optmizier)
    return gan_model


def make_patchs(imgs,crop_num=1):
    dataset = []
    for img in imgs:
        shape = img.shape
        Points_x = np.random.randint(0, shape[0] - patch_size, crop_num)
        Points_y = np.random.randint(0, shape[1] - patch_size, crop_num)
        for i in range(crop_num):
            dataset.append(img[Points_x[i]:Points_x[i] + patch_size, Points_y[i]:Points_y[i] + patch_size, :])
    return dataset

def make_array_patchs(imgs,crop_num=1,patch_size=32):
    if (len(imgs.shape) == 3):
        imgs = imgs.reshape((1,) + imgs.shape)
    #print(imgs.shape)
    Points_x = np.random.randint(0, imgs.shape[1] - patch_size, crop_num)
    Points_y = np.random.randint(0, imgs.shape[2] - patch_size, crop_num)
    for i in range(crop_num):
        dataset = imgs[:, Points_x[i]:Points_x[i] + patch_size, Points_y[i]:Points_y[i] + patch_size, :]
    return dataset

def img_normalize(img):
    return (img.astype('float32') - 127.5) / 127.5

def img_denormalize(img):
    return ((img + 1) * 127.5).astype(np.uint8)

def img_norm(img):
    img = img * 255
    img[img[:] > 255] = 255
    img[img[:] < 0] = 0
    img = img.astype('uint8')
    return img


def prepare_data(img_path, factor):
    imgs = os.listdir(img_path)
    img_path = [os.path.join(img_path, fname) for fname in imgs]
    label = [image.img_to_array(image.load_img(path, target_size=(384, 384))) for path in img_path]
    #label = [cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for path in img_path]
    #label = [img.astype('float32') / 255 for img in label]
    label = make_patchs(label)
    img = [cv2.resize(x, (patch_size // factor, patch_size // factor), interpolation=cv2.INTER_CUBIC) for x in label]
    label = [img_normalize(img) for img in label]
    img = [img_normalize(x) for x in img]
    return label, img


def train(epochs, input_dir, output_dir, predict_dir, batch_size):
    tf.config.experimental_run_functions_eagerly(True)
    train_dir = os.path.join(input_dir, 'Train')
    #validate_dir = os.path.join(input_dir, 'Validate')
    train_label, train_img = prepare_data(train_dir, 4)
    data_num = len(train_img)

    generator = get_generator(train_img[0].shape)
    discriminator = get_discriminator((32,32,3))
    #discriminator = get_discriminator(train_img[0].shape)
    #discriminator = get_discriminator(train_label[0].shape)

    loss = VGGLOSS(train_label[0].shape)
    adam = get_optimizer()
    generator.compile(loss=loss.vggloss, optimizer=adam)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)
    generator.load_weights('generator_epoch_200.h5')
    discriminator.load_weights('discriminator_epoch_200.h5')
    gan = get_gan(discriminator=discriminator, generator=generator, shape=train_img[0].shape, loss=loss.vggloss,
                  optmizier=adam)
    discriminator.summary()
    # generator.summary()

    batch_count = data_num // batch_size + (data_num % batch_size != 0)
    print('batch_count: %f\n' % batch_count)
    for epoch in range(1, epochs + 1):
        start = time.time()
        for index in range(batch_count):
            # print(index)
            train_img_batch = train_img[index * batch_size: (index + 1) * batch_size]
            train_label_batch = train_label[index * batch_size: (index + 1) * batch_size]
            num = len(train_img_batch)
            train_img_batch = np.array(train_img_batch)
            train_label_batch = np.array(train_label_batch)

            fake_img = make_array_patchs(generator.predict(train_img_batch))
            true_img = make_array_patchs(train_label_batch)
            true_target = np.ones(num, dtype=np.float32) - np.random.random_sample(num) * 0.2
            false_target = np.zeros(num, dtype=np.float32) + np.random.random_sample(num) * 0.2

            discriminator.trainable = True
            d_loss_1 = discriminator.train_on_batch(true_img, true_target)
            d_loss_2 = discriminator.train_on_batch(fake_img, false_target)
            d_loss = 0.5*(d_loss_1 + d_loss_2)

            misleading_target = np.ones((num)) - np.random.random_sample(num) * 0.2
            discriminator.trainable = False
            a_loss = gan.train_on_batch(train_img_batch, [train_label_batch, misleading_target])
            #print('discriminator_loss:{},  gan_loss: {}\n'.format(d_loss,a_loss))
        cost = time.time() - start
        print('<' + '=' * 15, 'Epoch %d' % epoch, '=' * 15 + '>  ')
        print('discriminator_loss:{},  gan_loss: {},   time: {}\n'.format(d_loss, a_loss, cost))
        if (epoch % 100 == 0):
            discriminator.save('discriminator_epoch_{}.h5'.format(epoch))
            generator.save('generator_epoch_{}.h5'.format(epoch))

input_dir = 'E:\\anaconda3\\envs\\tensorflow\\datasets\\SRCNN_Trainning'
output_dir = 'E:\\anaconda3\\envs\\tensorflow\\datasets\\SRCNN_Trainning\\result'
predict_dir = 'E:\\anaconda3\\envs\\tensorflow\\datasets\\SRCNN_Trainning\\Test\\set5\\butterfly_GT.bmp'
train(1000,input_dir=input_dir,output_dir=output_dir,predict_dir=predict_dir,batch_size=2)