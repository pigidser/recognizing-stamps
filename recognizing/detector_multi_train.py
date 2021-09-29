# Import Libraries
import dlib
import glob
import cv2
import os
import sys
import  time
import numpy as np
import matplotlib.pyplot as plt
import shutil

from detector_multi_utils import *

exp_num = 0

detector_directory = "./content234/"
img_directory = detector_directory + "images"
box_directory = detector_directory + "labels"

svm_parameter_c = 11

use_difficult = 1

print()
print("Training decoders for all labels defined for images")
print(f"Image folder: {img_directory}")
print(f"Labeled data folder: {box_directory}")
print(f"Use difficult cases: {use_difficult}")

### Data processing
data = get_data(box_directory, use_difficult)
print(f"Total labeled files: {len(data)}")

label_names = get_labels(data)
label_names = [x for x in label_names if x not in ['nakl-otpusk', 'transp-nakl']]
print(f"Found unique labels:")
print(label_names)

data_label = dict()

for label in label_names:
    data_label[label] = get_data_label(img_directory, data, label)
    print(f"Total labeled files for {label} label : {len(data_label[label])}")    

percent = 0.8

all_results = []

for label in label_names:

    detector_name = 'detector_' + label
    print()
    print()
    print(f"Training of detector '{detector_name}'")
    data = data_label[label]
    d_files = [tuple_value[0] for tuple_value in data.values()]
    d_data = [tuple_value[1] for tuple_value in data.values()]

    # How many examples make 80%.
    split = int(len(data) * percent)

    # Seperate the images and bounding boxes in different lists.
    images = [x[0] for x in d_data]
    bounding_boxes = [x[1] for x in d_data]

    ### simple_object_detector_training_options
    # http://dlib.net/python/index.html#dlib.simple_object_detector_training_options

    # Initialize object detector Options
    options = dlib.simple_object_detector_training_options()

    # options.detection_window_size = 6400 # default
    if label == 'mars-stamp':
        options.detection_window_size = 14400
    else:
        options.detection_window_size = 10000

    # Disabling the horizontal flipping. 
    options.add_left_right_image_flips = False

    # Set the c parameter of SVM equal to 5
    # A bigger C encourages the model to better fit the training data, it can lead to overfitting.
    # So set an optimal C value via trail and error.
    options.C = svm_parameter_c

    options.upsample_limit = 0

    options.num_threads = 4

    # Note the start time before training.
    st = time.time()

    # You can start the training now
    detector = dlib.train_simple_object_detector(images[:split], bounding_boxes[:split], options)

    # print(f"Trainig completed with {dlib.num_separable_filters(detector)} separable filters")
    # detector = dlib.threshold_filter_singular_values(detector, 0.05)
    # print(f"After threshold filter left {dlib.num_separable_filters(detector)} separable filters")

    # Print the Total time taken to train the detector
    print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

    ### Save the trained detector
    detector_filename = os.path.join(os.getcwd(), detector_directory, detector_name + '_d_' + str(use_difficult) + '_exp' + str(exp_num) + '.svm')
    if os.path.isfile(detector_filename):
        os.remove(detector_filename)
    detector.save(detector_filename)

    # Training part validation
    results = dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)
    train_precision = round(results.precision, 3)
    train_recall = round(results.recall, 3)
    print(f"Training Metrics: precision {train_precision}, recall {train_recall}")

    print("Train indexes")
    print(d_files[:split])

    # Test part validation
    results = dlib.test_simple_object_detector(images[split:], bounding_boxes[split:], detector)
    test_precision = round(results.precision, 3)
    test_recall = round(results.recall, 3)
    print(f"Testing Metrics: precision {test_precision}, recall {test_recall}")

    print("Test indexes")
    print(d_files[split:])

    all_results.append({'label': label, 'C': options.C, 'window_size': options.detection_window_size,
        'train_precision': train_precision, 'train_recall': train_recall,
        'test_precision': test_precision, 'test_recall': test_recall})


print(all_results)
