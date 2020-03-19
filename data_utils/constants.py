"""
module with predefined common constants related to data manipulation
"""

# dict defining indoor videos by tags
INDOOR = {'Bedroom,': '02_58j',
          'Bathroom,': '01j2bj',
          'Classroom,': '04hyxm',
          'Office': '021sj1'}

# dict defining outdoor videos by tags
OUTDOOR = {'Landscape,': '025s3q0',
           'Skyscraper,': '079cl',
           'Mountain,': '09d_r',
           'Beach': '0b3yr'}

# path/dir specific constants
INDOOR_DIR = 'indoor'
OUTDOOR_DIR = 'outdoor'

TRAIN_DATA = 'train'
VALIDATION_DATA = 'validation'
TEST_DATA = 'test'

# API url in order to get per category video hashes
VIDEOS_CATEGORY_API_URL = 'https://storage.googleapis.com/data.yt8m.org/2/j/v/{}'
# API url in order to get youtube video url
VIDEO_URL_API = 'https://storage.googleapis.com/data.yt8m.org/2/j/i/{}/{}'
# base url to youtube videos
VIDEO_BASE_PATH = 'http://youtube.com/watch?v={}'

# TRAIN, TEST, VALID split ratio
SPLIT_RATIO = (7, 2, 1)

IMAGE_SIZE_FOR_TRAINING = (224, 224)
IMAGE_SIZE_WITH_CHANNELS = IMAGE_SIZE_FOR_TRAINING + (3, )

# random seed for data generation
RANDOM_SEED = 42

CLASS_MAPPING = {0: 'indoor', 1: 'outdoor'}
