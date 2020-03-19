"""
Module for small custom models for experimenting.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from data_utils.constants import IMAGE_SIZE_WITH_CHANNELS


# "small model" idea based on the VGG network
small_model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=IMAGE_SIZE_WITH_CHANNELS),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(256, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(units=512),
    Dense(units=2, activation='softmax')
])
