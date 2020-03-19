"""
Script for frames extraction.
"""

import argparse
import glob
import logging
import os
import random
import shutil

import cv2
from tqdm import tqdm

from data_utils.common_functions import make_dirs
from data_utils.constants import (
    INDOOR_DIR, OUTDOOR_DIR, SPLIT_RATIO, TRAIN_DATA, TEST_DATA,
    VALIDATION_DATA, IMAGE_SIZE_FOR_TRAINING)


def parse_arguments():
    parser = argparse.ArgumentParser(description='extract frames')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input videos.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path extracted frames will be stored.')
    parser.add_argument('--start_delay', type=int, required=False,
                        default=0,
                        help='Number of omitted frames from the beginning.')
    parser.add_argument('--frame_offset', type=int, required=False,
                        default=1,
                        help='use every Nth frame.')
    return parser.parse_args()


def extract_frames(video_paths, output_path, start_delay, frame_offset):
    """
    Extract frames from videos

    :param video_paths: Path to downloaded videos
    :param output_path: Path where extracted frames should be stored
    :param start_delay: number of frames not to be used from the start of
                        the video
    :param frame_offset: use every Nth frame.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    count = 0
    for video_path in tqdm(video_paths):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        skipped = 0
        start = True
        while success:
            success, image = vidcap.read()
            if start and skipped < start_delay:
                skipped += 1
                continue
            if skipped < frame_offset:
                skipped += 1
                continue
            if image is None:
                continue
            # downsample images to the size for training
            # NOTE this would probably need a bit more investigation from
            # performance point of view
            frame = cv2.resize(image, IMAGE_SIZE_FOR_TRAINING)
            cv2.imwrite(os.path.join(output_path, f'frame{count}.jpg'), frame)
            count += 1
            skipped = 0
            start = False


def split_train_test_valid(input_images, output_path, category_dir,
                           split_ratio=SPLIT_RATIO):
    """
    Split data to train/test/validation

    :param input_images: Path to extracted frames
    :param output_path: Path where split data should be stored
    :param category_dir: Name of the category dir
    :param split_ratio: Ratio, (train, test, validation)
    """
    random.shuffle(input_images)
    train_images_count = len(input_images) // split_ratio[0]
    train_paths = input_images[:train_images_count]
    test_images_count = len(input_images) // split_ratio[1]
    test_paths = input_images[train_images_count:
                              train_images_count+test_images_count]
    validation_paths = input_images[train_images_count+test_images_count:]
    logging.info('creating train data dir')
    for path in tqdm(train_paths):
        file_path = os.path.join(output_path, TRAIN_DATA, category_dir,
                                 os.path.basename(path))
        make_dirs(os.path.dirname(file_path))
        shutil.move(path, file_path)
    logging.info('creating test data dir')
    for path in tqdm(test_paths):
        file_path = os.path.join(output_path, TEST_DATA, category_dir,
                                 os.path.basename(path))
        make_dirs(os.path.dirname(file_path))
        shutil.move(path, file_path)
    logging.info('creating validation data dir')
    for path in tqdm(validation_paths):
        file_path = os.path.join(output_path, VALIDATION_DATA, category_dir,
                                 os.path.basename(path))
        make_dirs(os.path.dirname(file_path))
        shutil.move(path, file_path)


def main():
    args = parse_arguments()
    output_path = args.output_path
    indoor_images = os.path.join(output_path, INDOOR_DIR)
    outdoor_images = os.path.join(output_path, OUTDOOR_DIR)
    # extract indoor video frames
    indoor_input_path = os.path.join(args.input_path, INDOOR_DIR)
    indoor_video_paths = glob.glob(os.path.join(indoor_input_path, '*.mp4'))
    extract_frames(indoor_video_paths, indoor_images, args.start_delay,
                   args.frame_offset)
    # extract outdoor video frames
    outdoor_input_path = os.path.join(args.input_path, OUTDOOR_DIR)
    outdoor_video_paths = glob.glob(os.path.join(outdoor_input_path, '*.mp4'))
    extract_frames(outdoor_video_paths, outdoor_images, args.start_delay,
                   args.frame_offset)
    # split data to train/test/valid indoor
    images_paths = glob.glob(os.path.join(indoor_images, '*.jpg'))
    split_train_test_valid(images_paths, output_path, category_dir=INDOOR_DIR)
    # split data to train/test/valid outdoor
    images_paths = glob.glob(os.path.join(outdoor_images, '*.jpg'))
    split_train_test_valid(images_paths, output_path, category_dir=OUTDOOR_DIR)


if __name__ == '__main__':
    main()
