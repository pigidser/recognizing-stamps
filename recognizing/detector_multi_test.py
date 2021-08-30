# Import Libraries
import dlib
import glob
import cv2
import os
import sys
import  time
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil
import xml.etree.ElementTree as ET

from detector_multi_utils import *

detector_directory = "./content2/"
img_directory = detector_directory + "images"

use_difficult = 1

def run_detection(filename):

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    scale_factor = 1.0

    # Content 2
    detector_mars_stamp = dlib.simple_object_detector(os.path.join(detector_directory, 'detector_mars-stamp' + '_d_' + str(use_difficult) + '.svm'))
    detector_another_stamp = dlib.simple_object_detector(os.path.join(detector_directory, 'detector_another-wide-stamp' + '_d_' + str(use_difficult) + '.svm'))
    detector_square_stamp = dlib.simple_object_detector(os.path.join(detector_directory, 'detector_square-stamp' + '_d_' + str(use_difficult) + '.svm'))

    detectors = [detector_mars_stamp, detector_another_stamp, detector_square_stamp]
    names = ['Mars', 'Another ', 'Square']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    shifts = [-23, -13, -3]
 
    img = cv2.imread(filename) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    res = apply_transform(img)

    [detections, confidences, detector_idxs] = \
        dlib.fhog_object_detector.run_multiple(detectors, res, upsample_num_times=0)

    print(filename)
    print(detections)
    print(confidences)
    print(detector_idxs)

    for i in range(len(detections)):

        # Since we downscaled the image we will need to resacle the coordinates according to the original image.
        x1 = int(detections[i].left() * scale_factor )
        y1 = int(detections[i].top() * scale_factor )
        x2 = int(detections[i].right() * scale_factor )
        y2 = int(detections[i].bottom() * scale_factor )

        print(x1, y1, x2, y2)

        # Draw the bounding box
        if confidences[i] * 100 > 1:

            cv2.putText(img, '{}: {:.2f}%'.format(names[detector_idxs[i]], confidences[i]*100), (x1 - 25, y2 + shifts[detector_idxs[i]]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[detector_idxs[i]], 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[detector_idxs[i]], 2 )

    cv2.imshow('frame', np.hstack([gray, res]))
    while True:
    
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.imshow('frame', img)

    while True:
    
        ch = cv2.waitKey(5)
        if ch == 27:
            break


test_files = ['20210812135430836-page-6.xml', '20210812135430836-page-8.xml', '20210812135517710-page-0.xml', '20210812135517710-page-1.xml', '20210812135517710-page-2.xml', '20210812135517710-page-3.xml', '20210812135517710-page-7.xml', '20210812135517710-page-9.xml', '20210812135619453-page-0.xml', '20210812135619453-page-10.xml', '20210812135619453-page-12.xml', '20210812135619453-page-14.xml', '20210812135619453-page-2.xml', '20210812135619453-page-4.xml', '20210812135619453-page-8.xml', '20210813182333055-page-10.xml', '20210813182333055-page-14.xml', '20210813182333055-page-16.xml', '20210813182333055-page-18.xml', '20210813182333055-page-2.xml', '20210813182333055-page-20.xml', '20210813182333055-page-22.xml', '20210813182333055-page-4.xml', '20210813182333055-page-6.xml', '20210813182333055-page-8.xml'] 

for test_file in test_files:
    filename = os.path.join(img_directory, test_file.split(".")[0] + '.png')

    run_detection(filename)
