"""
Script to download videos only for specified categories.

Unfortunately there is probably no simple way to do it directly from the yt8m
dataset website.
"""

import argparse
import logging
import os
import random
from ast import literal_eval

import requests
from pytube import YouTube
from pytube.exceptions import VideoUnavailable, RegexMatchError
from sqlitedict import SqliteDict
from tqdm import tqdm

from data_utils.common_functions import make_dirs
from data_utils.constants import (
    INDOOR, OUTDOOR, INDOOR_DIR, OUTDOOR_DIR, VIDEOS_CATEGORY_API_URL,
    VIDEO_URL_API, VIDEO_BASE_PATH, RANDOM_SEED)


def parse_arguments():
    parser = argparse.ArgumentParser(description='download videos')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path where downloaded videos will be stored.')
    parser.add_argument('--subsample', type=int, required=False, default=None,
                        help='videos subsample to be downloaded')
    return parser.parse_args()


def download_videos(tags_dict, output_path, subsample=None):
    """
    List urls of videos to indoor/outdoor according to tags.

    :param tags_dict: {tag: tag hash}
    :param output_path: path where videos should be stored
    :param subsample: number of samples per tag to be used
    :return: [urls]
    """
    make_dirs(output_path)
    # mechanism to be able to continue in the download
    cache = SqliteDict(os.path.join(output_path, 'cache.sqlite'),
                       autocommit=True)
    for tag, _hash in tags_dict.items():
        logging.info(f'downloading videos for tag: {tag}')
        response = requests.get(
            VIDEOS_CATEGORY_API_URL.format(_hash + '.js'))
        if response.status_code != 200:
            logging.warning(f'Unable to find videos for tag {tag}')
            continue
        _, video_hashes = literal_eval(response.text.strip('p;'))
        # possibility to work only with subsample data
        if subsample is not None and subsample < len(video_hashes):
            video_hashes = random.sample(video_hashes, k=subsample)
        for video_hash in tqdm(video_hashes):
            response = requests.get(VIDEO_URL_API.format(
                video_hash[:2], video_hash + '.js'))
            if response.status_code != 200:
                logging.warning(f'Unable to find video hashes for {video_hash}')
                continue
            _, youtube_hash = literal_eval(response.text.strip('i;'))
            url = VIDEO_BASE_PATH.format(youtube_hash)
            if url not in cache:
                try:
                    YouTube(url).streams.first().download(output_path)
                    cache[url] = True
                # NOTE KeyError is too general, but unfortunately thrown by
                # the API when missing streamingData
                except (VideoUnavailable, RegexMatchError, KeyError):
                    logging.warning(f'Unable to download video from {url}')


def main():
    random.seed(RANDOM_SEED)
    args = parse_arguments()
    download_videos(INDOOR, os.path.join(args.output_path, INDOOR_DIR),
                    args.subsample)
    download_videos(OUTDOOR, os.path.join(args.output_path, OUTDOOR_DIR),
                    args.subsample)


if __name__ == '__main__':
    main()
