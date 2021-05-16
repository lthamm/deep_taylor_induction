""" Test predicates in a hypothesis generated with this project,
    e.g. has(example, nose), therefore a dataset for the predicate needs
    to be generated not_has_nose, where the face is left unaltered except
    the negated predicate.

    All images have the ground_truth to be negative

    The examples for that dataset need to be placed in the
    config.predicate_picasso path
"""

import numpy as np

from models.vgg import VGGFinetune
from data_loaders.predicate_loader import PredicateTestLoader
import config


def eval_hypothesis(config):
    vgg = VGGFinetune().model

    # Get data flows
    loader = PredicateTestLoader(config)
    sets = loader.get_data_flows()

    # Load the pretrained model
    vgg = VGGFinetune().model

    for set in sets:

        print(f'[*] Evaluating {set["name"]} with {set["n_images"]} images')
        steps = (set['n_images'] // config.batch_size) + 1
        predictions = vgg.predict_generator(set['generator'], steps=steps)
        print(f'[+] Got predictions')

        rounded_predictions = np.rint(predictions)

        pos_predictions = np.sum(rounded_predictions == 1)
        neg_predictions = np.sum(rounded_predictions == 0)

        print(f'[*] Positive predictions: {pos_predictions}')
        print(f'[*] Negative predictions: {neg_predictions}')

        # images predicted to be negative, which by construction
        # of the dataset is the correct label for all images
        neg_percentage = (neg_predictions / len(rounded_predictions)) * 100

        # if wrong predictions exceed a certain threshold, the predicate
        # does not accurately describe the dataset
        if neg_percentage < config.validation_threshold:
            print(f'[-] Predicate failed with {neg_percentage}% \n')

        else:
            print(f'[-] Predicate passed with {neg_percentage}% \n')



if __name__ == "__main__":
    eval_hypothesis(config)
