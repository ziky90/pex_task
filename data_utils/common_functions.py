"""
Module of common functions.
"""

import os

import numpy as np

from data_utils.constants import (IMAGE_SIZE_FOR_TRAINING,
                                  IMAGE_SIZE_WITH_CHANNELS)


def make_dirs(path):
    """
    Create dirs when it does not exist.

    :param path: str -> path where directory should be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_image(image):
    """
    Perform image normalization.

    :param image: Pil image
    :return: normalized image as  a numpy array
    """
    # resize the image
    image = image.resize(IMAGE_SIZE_FOR_TRAINING)
    image_array = np.frombuffer(image.tobytes(), dtype=np.uint8)
    image_array = image_array.reshape(IMAGE_SIZE_WITH_CHANNELS)
    # rescale the image to (0, 1)
    return image_array / 255
