# Import Libraries
import dlib
import glob
import cv2
import os
import sys
import  time
import numpy as np
import matplotlib.pyplot as plt

from detector_multi_utils import *

detector_directory = "./content234/"
test_directory = "./test/images"
train_directory = "./content234/images"

use_difficult = 1

exp_prefix = '_best'

scale_factor = 1.0

# Load detectors and define the name, color and drawing offset
detector_mars_stamp = dlib.simple_object_detector(os.path.join(detector_directory, 'detector_mars-stamp' + '_d_' + str(use_difficult) + exp_prefix + '.svm'))
detector_another_stamp = dlib.simple_object_detector(os.path.join(detector_directory, 'detector_another-stamp' + '_d_' + str(use_difficult) + exp_prefix + '.svm'))
detector_square_stamp = dlib.simple_object_detector(os.path.join(detector_directory, 'detector_square-stamp' + '_d_' + str(use_difficult) + exp_prefix + '.svm'))

detectors = [detector_mars_stamp, detector_another_stamp, detector_square_stamp]
names = ['Mars', 'Another', 'Square']
colors = {"Mars": (0, 255, 0), "Another": (255, 0, 0), "Square": (0, 0, 255)}
shifts = {"Mars": -23, "Another": -13, "Square": -3}

# Number of detection of one image (from 1 to 4): detection one - initial image, others - rotated
number_detection_for_image = 4
rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
rotations = rotations[:number_detection_for_image]

# Distance between the centers of two detected objects when they are considered one object
tolerance = 50

show_detection_at_rotation = True


def run_detection(filename, rotation, show=False):
    """
    Apply detectors in one image with the selected rotation.

    """
    rotation_res = dict()

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    res = apply_transform(img)

    if not rotation is None:
        img = cv2.rotate(img, rotation)
        res = cv2.rotate(res, rotation)

    [detections, confidences, detector_idxs] = \
        dlib.fhog_object_detector.run_multiple(detectors, res, upsample_num_times=0)

    # print(filename)
    # print(detections)
    # print(confidences)
    # print(detector_idxs)

    for i in range(len(detections)):

        # Since we downscaled the image we will need to resacle the coordinates according to the original image.
        x1 = int(detections[i].left() * scale_factor )
        y1 = int(detections[i].top() * scale_factor )
        x2 = int(detections[i].right() * scale_factor )
        y2 = int(detections[i].bottom() * scale_factor )

        print(x1, y1, x2, y2)

        det_res = dict()
        det_res["detector"] = names[detector_idxs[i]]
        det_res["confidence"] = round(confidences[i], 3)
        det_res["coordinates"] = f"{x1} {y1} {x2} {y2}"
        det_res["center"] = f"{round(x1 + (x2-x1)/2)} {round(y1 + (y2-y1)/2)}"
        det_res["size"] = f"{res.shape[1]} {res.shape[0]}"
        rotation_res["detection_" + str(i)] = det_res

        if show:
            cv2.putText(img, '{}: {:.2f}%'.format(names[detector_idxs[i]], confidences[i]*100), (x1 - 25, y2 + shifts[names[detector_idxs[i]]]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[names[detector_idxs[i]]], 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[names[detector_idxs[i]]], 2 )

            cv2.setWindowTitle('frame', filename)
            cv2.imshow('frame', img)
            cv2.waitKey(0)

    return rotation_res


def get_init_coordinates(rotation, width, height, xt, yt):
    """
    Transpose x,y coordinates of a point in rotated image to coordinates of initial image.

    Parameters:
    -----------
    rotation - (str), how the image rotated:
        'rotation_0' - is not rotated,
        'rotation_1' - 90 degrees clockwise,
        'rotation_2' - 180 degrees,
        'rotation_3' - 90 degrees counter clockwise.
    width (int), height (int) - size of the rotated image.
    xt (int), yt (int) - x and y coordinates in the rotated image.

    Returns:
    -------
    tuple (int) - x and y coordinates of the point in the initial (not rotated) image.

    """
    n = rotation.split("_")[1]
    # width_n, height_n = map(int, size.split(" "))
    # x_n, y_n = map(int, center.split(" "))
    
    # Transpose the rotated coordinates to initial.
    if n == '0':
        x0, y0 = xt, yt
    elif n == '1':
        x0, y0 = yt, width - xt
    elif n == '2':
        x0, y0 = width - xt, height - yt
    elif n == '3':
        x0, y0 = height - yt, xt
    else:
        raise Exception(f"Rotation type '{rotation}' is not known.")

    return x0, y0


def get_init_corners(rotation, width, height, x1t, y1t, x2t, y2t):
    """
    Transpose top-left corner with x1t, y1t coordinates and
    bottom-right corner with x2t, y2t coordinates of an object
    in rotated image to coordinates of initial image.

    Parameters:
    -----------
    rotation - (str), how the image rotated:
        'rotation_0' - is not rotated,
        'rotation_1' - 90 degrees clockwise,
        'rotation_2' - 180 degrees,
        'rotation_3' - 90 degrees counter clockwise.
    width (int), height (int) - size of the rotated image.
    x1t, y1t, x2t, y2t (int) - top-left and bottom-right corners in the rotated image.

    Returns:
    -------
    tuple (int) - coordinates of corners in the initial (not rotated) image.

    """
    n = rotation.split("_")[1]
    
    # Transpose the rotated coordinates to initial.
    if n == '0':
        x1, y1, x2, y2 = x1t, y1t, x2t, y2t
    elif n == '1':
        x1, y1 = get_init_coordinates(rotation, width, height, x2t, y1t)
        x2, y2 = get_init_coordinates(rotation, width, height, x1t, y2t)
    elif n == '2':
        x1, y1 = get_init_coordinates(rotation, width, height, x2t, y2t)
        x2, y2 = get_init_coordinates(rotation, width, height, x1t, y1t)
    elif n == '3':
        x1, y1 = get_init_coordinates(rotation, width, height, x1t, y2t)
        x2, y2 = get_init_coordinates(rotation, width, height, x2t, y1t)
    else:
        raise Exception(f"Rotation type '{rotation}' is not known.")

    return x1, y1, x2, y2


def get_object_id(potential_objects, tolerance, x, y):
    """
    Return index of an object with the closest center. If distance
    between the centers of two detected objects less than 'tolerance',
    they are considered as one object.

    Parameters:
    -----------
    tolerance (int) - threshold value when two objects considered as one object.
    center (string) - center of the object.

    Returns:
    --------
    Index of object.
    
    """
    # x, y = map(int, center.split(" "))
    for i, obj in enumerate(potential_objects):
        x1, y1 = map(int, obj.split(" "))
        if (abs(x - x1) < tolerance) and (abs(y - y1) < tolerance):
            return i

    # An object not found. Let's add one and return last index.
    potential_objects.append(f"{x} {y}")

    return len(potential_objects) - 1


def get_mean_coordinates(coord_type, obj, x, y):

    if coord_type in obj:
        mean_coord = obj[coord_type]
        x_mean, y_mean = map(int, mean_coord.split(" "))
        x_mean, y_mean = (x_mean + x) // 2, (y_mean + y) // 2
    else:
        x_mean, y_mean = x, y
        
    return x_mean, y_mean


def find_possible_objects(im_result):

    # This dictionary will contain object with summary confidence that given in different rotations for each image 
    objects = dict()

    for img_det in im_result:

        potential_objects = list()

        img_name = img_det["name"]
        objects[img_name] = dict()

        print(img_name)
        
        for key in img_det:
            
            if "rotation" in key:

                rotation = key

                for det in img_det[rotation]:

                    obj = img_det[rotation][det]
                    object_type = obj["detector"]
                    object_confidence = obj["confidence"]
                    # Size of the image
                    width, height = map(int, obj["size"].split(" "))
                    # Get initial center coordinates
                    x0t, y0t = map(int, obj["center"].split(" "))
                    x0, y0 = get_init_coordinates(rotation, width, height, x0t, y0t)
                    # Get initial coordinates of left-top and bottom-right corners
                    x1t, y1t, x2t, y2t = map(int, obj["coordinates"].split(" "))
                    x1, y1, x2, y2 = get_init_corners(rotation, width, height, x1t, y1t, x2t, y2t)

                    print(f"{rotation} {object_type}, conf {object_confidence}, center {x0}, {y0}, top-left {x1}, {y1}, bottom-right {x2}, {y2}")
                    
                    # Get object id by coordinates of the center
                    obj_id = get_object_id(potential_objects, tolerance, x0, y0)

                    obj_conf_total = object_confidence

                    if obj_id in objects[img_name]:
                        
                        if object_type in objects[img_name][obj_id]["confidences"]:
                            conf = objects[img_name][obj_id]["confidences"][object_type]
                            obj_conf_total = conf + object_confidence

                        if object_type in objects[img_name][obj_id]["counts"]:
                            count = objects[img_name][obj_id]["counts"][object_type]
                        else:
                            count = 0
                    else:
                        objects[img_name][obj_id] = dict()
                        objects[img_name][obj_id]["confidences"] = dict()
                        objects[img_name][obj_id]["counts"] = dict()
                        count = 0
                    
                    # Total confidence
                    objects[img_name][obj_id]["confidences"][object_type] = obj_conf_total
                    # Counting of appearing
                    objects[img_name][obj_id]["counts"][object_type] = count + 1

                    # Get average coordinates of the center
                    x0_mean, y0_mean = get_mean_coordinates("center", objects[img_name][obj_id], x0, y0)
                    objects[img_name][obj_id]["center"] = f"{x0_mean} {y0_mean}"
                    # Get average coordinates of the top-left corner
                    x1_mean, y1_mean = get_mean_coordinates("top-left", objects[img_name][obj_id], x1, y1)
                    objects[img_name][obj_id]["top-left"] = f"{x1_mean} {y1_mean}"
                    # Get average coordinates of the bottom-right corner
                    x2_mean, y2_mean = get_mean_coordinates("bottom-right", objects[img_name][obj_id], x2, y2)
                    objects[img_name][obj_id]["bottom-right"] = f"{x2_mean} {y2_mean}"

    return objects


detection_results = list()

test_files = ['48-page-48.xml', '5-page-10.xml', '51-page-14.xml', '51-page-46.xml'] # ['48-page-48.xml', '48-page-5.xml', '48-page-7.xml', '48-page-8.xml', '49-page-10.xml', '49-page-12.xml', '49-page-16.xml', '49-page-18.xml', '49-page-2.xml', '49-page-21.xml', '49-page-23.xml', '49-page-24.xml', '49-page-26.xml', '49-page-3.xml', '49-page-30.xml', '49-page-36.xml', '49-page-38.xml', '49-page-4.xml', '49-page-40.xml', '49-page-42.xml', '49-page-43.xml', '49-page-45.xml', '49-page-48.xml', '49-page-6.xml', '49-page-7.xml', '5-page-10-2.xml', '5-page-10.xml', '5-page-13.xml', '5-page-14.xml', '5-page-17.xml', '5-page-2.xml', '5-page-5.xml', '5-page-6.xml', '5-page-7.xml', '5-page-9.xml', '50-page-1.xml', '50-page-2.xml', '50-page-3.xml', '50-page-4.xml', '50-page-5.xml', '50-page-6.xml', '50-page-7.xml', '50-page-8.xml', '51-page-10.xml', '51-page-14.xml', '51-page-16.xml', '51-page-20.xml', '51-page-22.xml', '51-page-24.xml', '51-page-26.xml', '51-page-28.xml', '51-page-32.xml', '51-page-33.xml', '51-page-36.xml', '51-page-38.xml', '51-page-40.xml', '51-page-42.xml', '51-page-44.xml', '51-page-45.xml', '51-page-46.xml', '51-page-48.xml', '51-page-49.xml', '51-page-51.xml', '51-page-52.xml', '51-page-55.xml', '51-page-6.xml', '51-page-8.xml', '6-page-2.xml', '6-page-4.xml', '6-page-6.xml', '6-page-8.xml', '7-page-10.xml', '7-page-12.xml', '7-page-15.xml', '7-page-2.xml', '7-page-6.xml', '7-page-8.xml', '8-page-10.xml', '8-page-13.xml', '8-page-14.xml', '8-page-17.xml', '8-page-18.xml', '8-page-2.xml', '8-page-21.xml', '8-page-22.xml', '8-page-25.xml', '8-page-26.xml', '8-page-28.xml', '8-page-3.xml', '8-page-30.xml', '8-page-31.xml', '8-page-32.xml', '8-page-34.xml', '8-page-35.xml', '8-page-37.xml', '8-page-5.xml', '8-page-6.xml', '8-page-7.xml', '8-page-9.xml', '9-page-10.xml', '9-page-14.xml', '9-page-18.xml', '9-page-2.xml', '9-page-22.xml', '9-page-26.xml', '9-page-30.xml', '9-page-6.xml']

if not isinstance(test_files, list):
    test_files = os.listdir(test_directory)
elif len(test_files) == 0:
    test_files = os.listdir(test_directory)


cv2.namedWindow('frame', cv2.WINDOW_NORMAL)


print()
print("Recognizing started")


for test_file in test_files:

    filename = os.path.join(train_directory, test_file.split(".")[0] + '.png')
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        continue

    im_result = dict()
    im_result["name"] = test_file


    for i, rotation in enumerate(rotations):
        rotation_res = run_detection(filename, rotation, show_detection_at_rotation)
        im_result["rotation_" + str(i)] = rotation_res

    detection_results.append(im_result)

    
    objects = find_possible_objects([im_result])

    print(objects)

    for obj_img in objects:

        img = cv2.imread(filename)

        for obj_id in objects[obj_img]:

            obj_names = list(objects[obj_img][obj_id]["confidences"].keys())
            obj_confs = list(objects[obj_img][obj_id]["confidences"].values())
            obj_counts = list(objects[obj_img][obj_id]["counts"].values())
            center = objects[obj_img][obj_id]["center"]
            x0, y0 = map(int, center.split(" "))
            top_left = objects[obj_img][obj_id]["top-left"]
            x1, y1 = map(int, top_left.split(" "))
            bottom_right = objects[obj_img][obj_id]["bottom-right"]
            x2, y2 = map(int, bottom_right.split(" "))

            m = np.argmax(obj_confs)
            object_type = obj_names[m]
            object_conf = obj_confs[m]
            object_count = objects[obj_img][obj_id]["counts"][object_type]
            object_conf = round(object_conf/object_count, 2)
            print(f"Image {obj_img}, #{obj_id} is {obj_names[m]} at position {center} confidence {object_conf}")
            
            if (object_count >= 2 and object_conf > 0.1) or (object_count >= 1 and object_conf > 0.3):

                # Draw the bounding box
                cv2.putText(img, '{}: {:.2f}%'.format(object_type, object_conf * 100), (x1 - 25, y2 + shifts[object_type]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[object_type], 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[object_type], 2 )
            else:
                print("The object is not passed the checks.")

        cv2.setWindowTitle('frame', filename + " - final result")
        # cv2.imshow('frame', np.hstack([gray, res]))

        cv2.imshow('frame', img)

        cv2.waitKey(0)


# for res in detection_results:
#     print(res)