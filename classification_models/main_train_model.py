"""
Script for training the CNN model, either from scratch or as fine-tuning.
"""

import argparse
import logging
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classification_models.custom_models import small_model
from data_utils.common_functions import make_dirs
from data_utils.constants import (TRAIN_DATA, VALIDATION_DATA, INDOOR_DIR,
                                  OUTDOOR_DIR, IMAGE_SIZE_FOR_TRAINING,
                                  TEST_DATA, IMAGE_SIZE_WITH_CHANNELS)

# Model config variables
BATCH_SIZE = 64
VERTICAL_FLIP = False
EPOCHS = 20


def parse_arguments():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to extracted data frames.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path where trained model should be stored.')
    parser.add_argument('--model', type=str, required=True,
                        choices=('simple', 'resnet', 'mobilenet', 'densenet'),
                        help='CNN type to be used.')
    parser.add_argument('--use_weights', dest='use_weights', action='store_true',
                        help='Use pre-trained weights from imagenet.')
    parser.add_argument('--batch_size', type=int, required=False,
                        default=BATCH_SIZE, help='Batch size for training.')
    return parser.parse_args()


def prepare_dataset(data_path, batch_size):
    """
    Prepare dataset for training.

    :param data_path: path to the data with subdirs (train, test, valid)
    :param batch_size: desired batch size for training
    :return: train data generator, val data generator, train samples, val samples
    """
    # Prepare data generator
    train_dir = os.path.join(data_path, TRAIN_DATA)
    val_dir = os.path.join(data_path, VALIDATION_DATA)
    # NOTE normalize images to (0, 1) usually works good enough for a quick
    # prototype, but mean centering and std normalization might even improve the
    # model performance.
    train_images_generator = ImageDataGenerator(rescale=1./255,
                                                vertical_flip=VERTICAL_FLIP)
    val_images_generator = ImageDataGenerator(rescale=1./255)
    train_data_generator = train_images_generator.flow_from_directory(
        batch_size=batch_size, directory=train_dir, shuffle=True,
        target_size=IMAGE_SIZE_FOR_TRAINING, class_mode='categorical')
    val_data_generator = val_images_generator.flow_from_directory(
        batch_size=batch_size, directory=val_dir, class_mode='categorical',
        target_size=IMAGE_SIZE_FOR_TRAINING)
    # compute data statistics
    train_indoor_dir = os.path.join(train_dir, INDOOR_DIR)
    train_outdoor_dir = os.path.join(train_dir, OUTDOOR_DIR)
    val_indoor_dir = os.path.join(val_dir, INDOOR_DIR)
    val_outdoor_dir = os.path.join(val_dir, OUTDOOR_DIR)
    indoor_train_count = len(os.listdir(train_indoor_dir))
    outdoor_train_count = len(os.listdir(train_outdoor_dir))
    indoor_val_count = len(os.listdir(val_indoor_dir))
    outdoor_val_count = len(os.listdir(val_outdoor_dir))
    train_total = indoor_train_count + outdoor_train_count
    val_total = indoor_val_count + outdoor_val_count
    logging.info(f'Indoor train examples: {indoor_train_count}, outdoor train '
                 f'examples: {outdoor_train_count}, total: f{train_total}')
    logging.info(f'Indoor valid examples: {indoor_val_count}, outdoor valid '
                 f'examples: {outdoor_val_count}, total: {val_total}')
    return train_data_generator, val_data_generator, train_total, val_total


def build_model(model_name, use_weights):
    """
    Prepare the model that should be trained.

    :param model_name: Name of the model to be trained
    :param use_weights: Use pre-trained weights bool
    :return: tf.keras.Model
    """
    if use_weights:
        weights = 'imagenet'
    else:
        weights = None
    if model_name == 'simple':
        model = small_model
    elif model_name == 'mobilenet':
        model = MobileNetV2
        model = model(input_shape=IMAGE_SIZE_WITH_CHANNELS,
                      weights=weights, include_top=False)
        for layer in model.layers:
            layer.trainable = False
        x = model.output
        x = GlobalAveragePooling2D()(x)
        # NOTE removed regularization in order to be able to train at least
        # something that works given limited computational resources.
        x = Dense(units=2, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=x)
    else:
        raise NotImplementedError(f'Model not supported: {model_name}')
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy', Precision(), Recall()])
    model.summary()
    return model


def train_model(model, train_data_generator, val_data_generator, train_samples,
                val_samples, output_path):
    """
    Perform model entire training.

    :param model: Built tf.keras.model
    :param train_data_generator: Generator over training data
    :param val_data_generator: Generator over validation data
    :param train_samples: Number of training samples
    :param val_samples: Number of validation samples
    :param output_path: Path where the model should be stored
    :return: History of the model training
    """
    history = model.fit(
        train_data_generator,
        steps_per_epoch=train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data_generator,
        validation_steps=val_samples // BATCH_SIZE
    )
    make_dirs(output_path)
    model.save(output_path)
    return history


def evaluate_model(model, history, data_path):
    """
    Perform model evaluation.

    :param model: trained model
    :param history: training history
    :param data_path: path to the dir with data (train, test, valid)
    """
    test_dir = os.path.join(data_path, TEST_DATA)
    test_images_generator = ImageDataGenerator(rescale=1. / 255)
    test_data_generator = test_images_generator.flow_from_directory(
        batch_size=BATCH_SIZE, directory=test_dir,
        target_size=IMAGE_SIZE_FOR_TRAINING, class_mode='categorical')
    evaluation = model.evaluate(test_data_generator)
    print(f'test data accuracy: {evaluation[1]}, precision: {evaluation[2]},'
          f'recall: {evaluation[3]}')
    print(f'validation data accuracy: {history.history["val_accuracy"][-1]}, '
          f'precision: {history.history["val_precision"][-1]}, '
          f'recall: {history.history["val_recall"][-1]}')
    print(f'train data accuracy: {history.history["accuracy"][-1]}, '
          f'precision: {history.history["precision"][-1]}, '
          f'recall: {history.history["recall"][-1]}')


def main():
    args = parse_arguments()
    model = args.model
    use_weights = args.use_weights
    data_path = args.data_path
    batch_size = args.batch_size
    output_path = args.output_path
    if model == 'simple' and use_weights:
        raise Exception('simple model can not be used with pretrained weights'
                        'from imagenet.')
    train_data_generator, val_data_generator, train_samples, val_samples = \
        prepare_dataset(data_path, batch_size)
    model = build_model(model, use_weights)
    history = train_model(model, train_data_generator, val_data_generator,
                          train_samples, val_samples, output_path)
    evaluate_model(model, history, data_path)


if __name__ == '__main__':
    main()
