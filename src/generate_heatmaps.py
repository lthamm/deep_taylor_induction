"""Select images to generate the ilp background knowledge based on.
    1. Load test data and classify using the model
    2. Sample those images the model is most uncertain about
    3. Heatmap the sampled images and store outputs

    The output filenames have the following naming format:
    neg     _   neg     _   pic_00046.png
    pred.   -   truth   -   base filename
 """

import os
import argparse

import keras
from imutils import paths

from ilp import sample_data
from heatmapping.deep_taylor import DeepTaylor
from models.vgg import VGGFinetune
from data_loaders.picasso_loader import PicassoLoader
import config

parser = argparse.ArgumentParser(description='Select images and create heatmaps')

parser.add_argument('--n', default=50, type=int,
                    help=('Number of images to select for heatmapping'
                          'per class (positive / negative).'
                          'Has to be divideable by 10.')
                    )

parser.add_argument('--only_correct', default=False, type=bool,
                    help=('Decide wether to include only correctly classified'
                          'images in the output folders pos and neg.'
                          'If this is set to true, pos will be true positives'
                          'and negative will be true negatives.'
                          'Idepently of this argument, there will be folders'
                          'containing all false positives and false negatives.')
                    )


def load_data():
    """Load the test data to select from"""
    training, validation, test, info = PicassoLoader().get_data_flows()
    return test, info


def selection_generator():
    """Flow for the selected images"""

    augment = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.vgg16.preprocess_input
    )

    data = augment.flow_from_directory(
        config.ilp_img_path,
        class_mode='binary',
        target_size=(224, 224),
        color_mode='rgb',
        shuffle=False,
        batch_size=10)
    # in case of a filenumber % 10 = 0, an even number of steps for the
    #   analyzer can be used with this

    return data


def n_selected_images():
    """Return number of images"""
    try:
        return len(list(paths.list_images(config.ilp_img_path)))
    except Exception as e:
        # might fail in case dir has not been created
        print('[!] Failed checking for existing images')
        print(e)
        return 0  # in that case there are no images yet


if __name__ == "__main__":

    args = parser.parse_args()

    vgg = VGGFinetune()

    # check if images already exist
    n_images = n_selected_images()

    if n_images < 50:
        # Load data, classify with the model, and sample
        test, info = load_data()
        sample_data.sample(test,
                           vgg,
                           info,
                           n=args.n,
                           only_correct=args.only_correct)

    # Heatmaps
    steps = (n_images // 10)
    data = selection_generator()
    data.reset()
    analyzer = DeepTaylor(vgg.model, data)
    analysis = analyzer.analyze(steps)
    hmap_path = os.path.sep.join([config.ilp_path, 'deep_taylor_images'])
    analyzer.save_images(hmap_path)
