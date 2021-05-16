import os

import keras
import numpy as np
from imutils import paths

import config


class PredicateTestLoader():
    """ Loads the customized Picasso Dataset for hypothesis test from harddrive,
        creates generators and tensorflow flows
    """

    def __init__(self, config):

        # Get all the folders in predicates_path
        # each of which has data for one predicate to test
        self.sets = []

        for item in os.listdir(config.predicate_picasso):
            full_path = os.path.join(config.predicate_picasso, item)

            # Use only the folder
            if not os.path.isfile(full_path):

                dir = full_path  # directory containg samples
                name = os.path.basename(full_path)  # e.g. not_has_nose
                n_images = len(list(paths.list_images(dir)))

                set = {'name': name,
                        'path': dir,
                        'n_images': n_images,
                        'generator': None}

                self.sets.append(set)

        self.create_augmentations()

    def create_augmentations(self):

        # Augmentation for the validation data with only mean substraction
        val_augment = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.vgg16.preprocess_input
        )

        # ImageNet mean subtraction (in RGB order) and set the
        # might be unneccesarry
        IMAGENET_MEAN = np.array([103.939, 116.779, 123.68], dtype="float32")
        val_augment.mean = IMAGENET_MEAN
        self.val_augment = val_augment


    def get_data_flows(self):
        """Return the datasets for the hypothesis validation

        Return
        ------
        sets : list
            List of dictionaries with name, path, number of images and
            generator (flow) for each hypothesis validation set
        """

        for set in self.sets:

            generator = self.val_augment.flow_from_directory(
                    set['path'],
                    class_mode='binary',
                    target_size=(224, 224),
                    color_mode='rgb',
                    shuffle=False,
                    batch_size=config.batch_size)

            set['generator'] = generator

        return self.sets
