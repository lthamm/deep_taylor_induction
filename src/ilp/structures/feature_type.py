from enum import Enum

class FeatureType(Enum):
    """ Enumerator for feature constants """
    left_eye = 0
    right_eye = 1
    nose = 2
    mouth = 3
    face_frame = 4

    def __str__(self):
        # return the name of the own feature type
        return FeatureType(self).name
