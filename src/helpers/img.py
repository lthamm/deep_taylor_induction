import os


def path_to_name(img):
    """Images from the training and test data come from a directory structe
    that looks like this:
        test/neg/filename.png
        test/pos/filename.png

    This provides information about the associated class for keras.
    Yet it becomes obselete after classification or for heatmapping,
    because we can have false positives as well as false negatives.
    While we still want to keep the original information associated with the
    images, the directory structure is not neccesarry anymore and makes
    further processing more difficult, so the desired output is:
        neg_filname.png
        pos_filename.png
    """

    return os.path.dirname(img) + '_' + os.path.basename(img)


def name_to_path(img, origin):
    """Given a filename return the original path

    Parameters
    ----------
    origin : str
        Base path the img is orignally from

    """

    orig_file_parts = img.split('_')[1:]

    category = orig_file_parts[-3]
    filename = orig_file_parts[-2]+'_'+orig_file_parts[-1]

    orig_file = os.path.sep.join([origin, category])
    orig_file = os.path.sep.join([orig_file, filename])

    return orig_file
