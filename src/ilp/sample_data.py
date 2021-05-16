""" Choose training examples for generating the ilp background knowledge

    1. Load the test set generator
    2. Make predictions on the test set
    3. Pick n samples per class closest to the descision boundary based on the
        predictions
    4. Copy the samples into a new folder

"""
import os
import shutil

import pandas as pd

import config
import helpers


def make_predictions(data, model, info):
    """Predict """

    # Predictions
    print('[*] Making predictions on test set, this may take a while')
    data.reset()
    steps = (info['n_test'] // config.batch_size) + 1
    predictions = model.predict_generator(data, steps=steps)
    print(f'[+] Got {len(predictions)} predictions')

    # Summary Information
    data.reset()
    filenames = data.filenames
    print(f'[+] Got {len(filenames)} filenames')

    labels = data.class_indices.items()
    print(labels)

    data.reset()
    # Get the ground truth labels
    ground_truth = data.classes
    print(f'[+] Got {len(ground_truth)} actual labels')

    # construct a dataframe that stores the results
    # filename | ground_truth | prediction | predicted_class
    df = pd.DataFrame()
    df['filename'] = filenames
    df['ground_truth'] = ground_truth
    df['prediction'] = predictions
    df['predicted_class'] = df['prediction'].apply(lambda x: round(x))

    print(df)

    # Store info as a pickle
    pickle_path = os.path.sep.join([config.ilp_path, 'pickles'])
    os.makedirs(pickle_path, exist_ok=True)
    df.to_pickle(os.path.sep.join([pickle_path, 'df.p']))

    return df


def pick_images(df, only_correct=False, n=50):
    """Pick the images closest to the descision boundary per class

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the predictions in the following format:
        filename | ground_truth | prediction | predicted_class

    only_correct : boolean
        Select only samples that have a correct prediction for the main
            output
            if only_correct == True:
                positive contains true positive and negative contains true
                negative

    n : int
        Number of images per class
    """

    # Make sure n is divideable by 10
    if n % 10 != 0:
        raise ValueError('n must be dividable by 10')

    print(f'[*] Picking images')

    # Selective for false outputs
    pred_0 = df['predicted_class'] == 0
    pred_1 = df['predicted_class'] == 1

    gt_0 = df['ground_truth'] == 0
    gt_1 = df['ground_truth'] == 1

    false_positive = df[pred_1 & gt_0]
    false_negative = df[pred_0 & gt_1]

    # General output
    # predictions that are correct
    if only_correct:
        df = df[df['ground_truth'] == df['predicted_class']]

    positive = df[df['predicted_class'] == 1]
    negative = df[df['predicted_class'] == 0]

    # Select only the n values closest to the descion boundary
    positive = positive.sort_values(by='prediction', ascending=True).head(n)
    negative = negative.sort_values(by='prediction', ascending=False).head(n)

    print(positive)
    print(negative)

    return (positive, negative, false_positive, false_negative)


def filename_lists(positive, negative, false_positive, false_negative):
    """Returns the filenames given dataframes as a tuple of lists"""
    return (positive['filename'].tolist(), negative['filename'].tolist(),
            false_positive['filename'].tolist(),
            false_negative['filename'].tolist())


def copy_images(selections, classes=['pos', 'neg', 'fp', 'fn']):
    """Copy the selected images into a new folder

    Parameters
    ----------
    selections : Tuple
        Tuple of lists of filenames in the following order:
        positive, negative, false positive and false negative

    classes : List
        List of strings for the classes that shall be saved
    """

    print('[*] Copying images')

    for images, image_class in zip(selections, classes):

        # Prepare paths
        path = os.path.sep.join([config.ilp_img_path, image_class])

        # Create the folders in case they don't exist
        os.makedirs(path, exist_ok=True)

        for img in images:

            filename = helpers.img.path_to_name(img)
            shutil.copy(os.path.sep.join([config.test, img]),
                        os.path.sep.join([path, filename])
                        )


def sample(test, vgg, info, n=50, only_correct=False):
    """ Main method for starting calling the sampling"""
    prediction_df = make_predictions(test, vgg.model, info)

    positive, negative,\
        false_positive, false_negative = pick_images(prediction_df, n=n,
                                                     only_correct=only_correct)

    # Returns the filenames as a tuples of lists
    selections = filename_lists(positive,
                                negative,
                                false_positive,
                                false_negative)
    copy_images(selections)
