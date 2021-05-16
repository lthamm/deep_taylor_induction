import os

import keras
import numpy as np
from imutils import paths

import config

class PicassoLoader():
    """ Loads the Picasso Dataset from harddrive, creates generators and
    tensorflow flows"""

    def __init__(self):

        # Construct path for the dataset
        self.train_path = os.path.sep.join([config.base_path, config.train])
        self.val_path = os.path.sep.join([config.base_path, config.val])
        self.test_path = os.path.sep.join([config.base_path, config.test])

        # Store statistics about the training data
        self.n_train = len(list(paths.list_images(self.train_path)))
        self.n_validation = len(list(paths.list_images(self.val_path)))
        self.n_test = len(list(paths.list_images(self.test_path)))

        self.info = {'n_train': self.n_train,
                'n_validation': self.n_validation,
                'n_test': self.n_test
        }

        self.create_augmentations()

    def create_augmentations(self):
        # Augmentation for the training data
        #   flips, rotates, zooms input data on the fly to increase input

        # Preprocessing function converts from RGB to BGR
        #   https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input
        train_augment = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.vgg16.preprocess_input,
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")


        # Augmentation for the validation data with only mean substraction
        val_augment = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.vgg16.preprocess_input
        )

        # ImageNet mean subtraction (in RGB order)
        # might be unneccesarry
        IMAGENET_MEAN = np.array([103.939, 116.779, 123.68], dtype="float32")
        train_augment.mean = IMAGENET_MEAN
        val_augment.mean = IMAGENET_MEAN

        self.train_augment = train_augment
        self.val_augment = val_augment

        return train_augment, val_augment # in case its neeeded


    def get_data_flows(self):
        # Load data from harddrive, so it doesnt have to be stored in ram
        #   offered by the flow_from_directory
        #   these are generators, yielding the relevant samples
        print('[+] Got training examples:')
        training = self.train_augment.flow_from_directory(
            self.train_path,
            class_mode='binary',
            target_size=(224, 224),
            color_mode='rgb',
            shuffle=True,
            batch_size=config.batch_size)

        print('[+] Got validation examples:')
        validation = self.val_augment.flow_from_directory(
            self.val_path,
            class_mode='binary',
            target_size=(224, 224),
            color_mode='rgb',
            shuffle=False,
            batch_size=config.batch_size)

        print('[+] Got test examples:')
        test = self.val_augment.flow_from_directory(
            self.test_path,
            class_mode='binary',
            target_size=(224, 224),
            color_mode='rgb',
            shuffle=False,
            batch_size=config.batch_size)

        return training, validation, test, self.info
