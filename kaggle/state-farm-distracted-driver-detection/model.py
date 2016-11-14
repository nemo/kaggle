from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras.utils.data_utils import get_file
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

class Model():
    """CNN model based on VGG16"""
    
    def __init__(self):
        self.FILE_PATH = 'http://www.platform.ai/models/'
        self.img_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
        self.create()
        self.get_imagenet_classes()
    

    def get_imagenet_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def pre_preprocess(self, x):
        x = x - self.img_mean
        return x[:, ::-1]
    
    def ConvBlock(self, layers, filters):
        for i in range(layers):
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(filters, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
    
    def create(self):
        model = self.model = Sequential()
        
        model.add(Lambda(self.pre_preprocess, input_shape=(3,224,224)))
        
        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))

        
    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    
    def finetune(self, batches):
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable=False
        
        model.add(Dense(batches.nb_class, activation='softmax', input_shape=(1000,)))
        
        for layer in model.layers[:-5]:
            layer.trainable=True

        model.compile(optimizer=RMSprop(lr=0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, batches, val_batches, nb_epoch=1):
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

    
    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)
    
