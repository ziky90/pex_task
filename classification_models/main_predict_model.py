"""
Module for prediction from the ML model.

NOTE: the module predicts only one image with normalization performed by
different code than used for training (which is not nice). For production
probably some unification would be needed.
"""
import argparse

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from data_utils.common_functions import normalize_image
from data_utils.constants import CLASS_MAPPING


def parse_arguments():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to image to be predicted.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model to be used.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_path = args.model_path
    image_path = args.image_path
    model = load_model(model_path)
    image = Image.open(image_path)
    normalized_image = normalize_image(image)
    predicted_probas = model.predict(np.array([normalized_image]))
    print(predicted_probas[0])
    print(CLASS_MAPPING[np.argmax(predicted_probas)])


if __name__ == '__main__':
    main()
