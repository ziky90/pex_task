"""
Unit tests for common_functions.py
"""
import unittest

from PIL import Image
import numpy as np

from data_utils.common_functions import normalize_image
from data_utils.constants import IMAGE_SIZE_WITH_CHANNELS


class TestNormalizeImage(unittest.TestCase):

    def test_normalize_image_zeros(self):
        """
        Test normalization on image with edge values of 0, no resize
        """
        array = np.tile(0, IMAGE_SIZE_WITH_CHANNELS).astype('uint8')
        image = Image.fromarray(array)
        normalized_image = normalize_image(image)
        np.testing.assert_array_equal(normalized_image, array)

    def test_normalize_image_ones(self):
        """
        Test normalization on image with edge values of 255, no resize.
        """
        array = np.tile(255, IMAGE_SIZE_WITH_CHANNELS).astype('uint8')
        image = Image.fromarray(array)
        normalized_image = normalize_image(image)
        expected = np.tile(1., IMAGE_SIZE_WITH_CHANNELS)
        np.testing.assert_array_equal(normalized_image, expected)

    def test_normalize_image_ones_resize(self):
        """
        Test normalization on image with edge values of 255 with resize.
        """
        test_image_size = (450, 320, 3)
        array = np.tile(255, test_image_size).astype('uint8')
        image = Image.fromarray(array)
        normalized_image = normalize_image(image)
        expected = np.tile(1., IMAGE_SIZE_WITH_CHANNELS)
        np.testing.assert_array_equal(normalized_image, expected)


if __name__ == '__main__':
    unittest.main()
