# Indoor / Outdoor image classification based on YouTube-8M dataset

This repository contains implementation of the Machine Learning technical challenge. Goal of the challenge is to train a model that is able to classify images between indoor and outdoor.

The repository contains several scripts that can be used independently, but the expected workflow is as follows:
1. Download videos with specific tags from YouTube
2. Extract desired frames from downloaded videos
3. (optional, but recommended) manual check / filtering of extracted frames
4. Model training (there are several options)
5. Inference from the model

## Download videos with specific tags from YouTube

For this purpose one can use script `data_utils/main_download_videos.py`.

Example usage:

`python data_utils/main_download_videos.py --output_path ~/data/yt8m/classified_videos --subsample 250`

In order to change tags to be downloaded, INDOOR and OUTDOOR configuration dictionaries within the `data_utils/constants.py`should be changed.

## Extract desired frames from downloaded videos
There exists script `data_utils/main_extract_frames.py` for this purpose.

Example usage:

`python data_utils/main_extract_frames.py --input_path ~/data/yt8m/classified_videos --output_path ~/data/yt8m/classified_videos_frames --start_delay 600 --frame_offset 200`

## Manual check / filtering of extracted frames (optional)
This task should be performed manually. Images that does not correspond classes outdoor / indoor should be manually removed from the training set in order to achieve higher quality model.

NOTE: for purposes of the assignment there was not enough time in order to perform this task.

## Model training

Currently there are two different models implemented. The implementation can be easily extended to different custom models or to any model from `tensorflow.keras.applications`.

Training related parameters can be set in `classification_models/main_train_model.py`.

### Simple CNN training from scratch

The simple model can easily be trained by command: `python classification_models/main_train_model.py --data_path ~/data/yt8m/classified_videos/frames --output_path ~/data/yt8m/classified_videos/model --model simple`

On my prepared dataset in particular I've achieved following performance:

```
test data accuracy: 0.734093427658081
validation data accuracy: 0.7484375238418579
train data accuracy: 0.8502604365348816
```

### MobileNet fine-tuning

MobileNet can be fine-tuned by the command: `python classification_models/main_train_model.py --data_path ~/data/yt8m/classified_videos/frames --output_path ~/data/yt8m/classified_videos/model_mobilenet --model mobilenet --use_weights`

On my data in particular I've achieved following performance:

```
test data accuracy: 0.8521515130996704
validation data accuracy: 0.8453124761581421
train data accuracy: 0.9227527976036072
```

### NOTE regarding the model evaluation

For purposes of the task only loss and accuracy are considered. Since the dataset for the training was balanced I didn't have to use other metrics. For case when the dataset will not be perfectly balanced there are prepared `precision` and `recall` metrics which can easily be extended towards `F1` or `F beta` in general.


## Inference from the model

For the inference from the model one can use following command: `python classification_models/main_predict_model.py --image_path ~/data/yt8m/classified_videos/frames/test/outdoor/frame1002.jpg --model_path ./model`

NOTE: model is stored using git lfs, for getting started with `git lfs` please see: https://git-lfs.github.com/

## Future work

1. I believe that paying more attention to the training data would probably significantly improve the model accuracy. There can possibly be deployed some manual filters to images that are corrupted and this can possibly be combined with manual filtering. Manual filtering can be simplified by active-learning approach.
2. There should be performed more experiments with various ML models. Trade-off between the accuracy and inference time should be considered. 
