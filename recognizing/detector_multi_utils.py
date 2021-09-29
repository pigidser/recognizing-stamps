import os
import dlib
import cv2
import numpy as np
import xml.etree.ElementTree as ET


def get_img_boxes(box_file_name, use_difficult):
    """
    Get image file name and all boxes for one box file.

    """
    data = dict()
    d = dict()
    tree = ET.parse(box_file_name)    
    root = tree.getroot()
    # children = root.getchildren()
    for tag1 in root.getchildren():
        if tag1.tag == 'filename':
            filename = tag1.text
        elif tag1.tag == 'object':
            for tag2 in tag1.getchildren():
                if tag2.tag == 'name':
                    label_name = tag2.text
                elif tag2.tag == 'difficult':
                    difficult = int(tag2.text)
                elif tag2.tag == 'bndbox':
                    for tag3 in tag2.getchildren():
                        if tag3.tag == 'xmin':
                            x1 = int(tag3.text)
                        elif tag3.tag == 'ymin':
                            y1 = int(tag3.text)
                        elif tag3.tag == 'xmax':
                            x2 = int(tag3.text)
                        elif tag3.tag == 'ymax':
                            y2 = int(tag3.text)
            if difficult == 0 or use_difficult == 1:
                if label_name in d.keys():
                    d[label_name].append(list([x1, y1, x2, y2]))
                else:
                    d[label_name] = list([[x1, y1, x2, y2]])
    data['filename'] = filename
    data['boxes'] = d
    return data


def get_data(box_directory, use_difficult):
    """
    Get data structure (image file name and boxes information) for all box files in a directory.

    """
    data = list()
    for box_name in os.listdir(box_directory):
        data.append((box_name, get_img_boxes(os.path.join(box_directory, box_name), use_difficult)))
    return data


def get_labels(data):
    """
    Get all unique labels that are presented in data structure.

    """
    labels = list()
    for box_info in data:
        if 'boxes' in box_info[1].keys():
            for label in box_info[1]['boxes']:
                labels.append(label)

    labels = tuple(set(labels))    
    return labels


def get_data_label(img_directory, data, label):
    """
    Get data structure for selected label that is used by particular detector.

    """
    data_label = dict()
    i = 0
    for box_info in data:
        box = list()
        # print(box_info[1].keys())
        if 'boxes' in box_info[1].keys():
            if label in box_info[1]['boxes']:
                boxes = box_info[1]['boxes'][label]
                if len(boxes) > 0:
                    for box in boxes:
                        # print(box_info[1]['filename'])
                        # print(box_info[1]['boxes'][label])
                        img = cv2.imread(os.path.join(img_directory, box_info[1]['filename']))
                        
                        res = apply_transform(img)

                        x1, y1, x2, y2  = box[0], box[1], box[2], box[3]

                        # side1, side2 = (x2 - x1), (y2 - y1)
                        # ratio = round(side1 / side2, 1)
                        # # if side1 > side2:
                        # #     ratio = round(side1 / side2, 1)
                        # # else:
                        # #     ratio = round(side2 / side1, 1)
                        # print(f"{box_info[1]['filename']} {ratio} {side1} {side2}")

                        dlib_box = [ dlib.rectangle(left=x1 , top=y1, right=x2, bottom=y2) ]
                
                        # Store the image and the box together
                        data_label[i] = (box_info[0], (res, dlib_box))
                        i += 1

    return data_label


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def apply_transform(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

    # Form the initial and final filter color, brightness and contrast.
    # Variant 1
    h_min = np.array((90, 15, 85), np.uint8)
    h_max = np.array((255, 255, 255), np.uint8)
    brightness, contrast = 5 - 127, 255 - 127
    # # Variant 2
    # h_min = np.array((0, 40, 70), np.uint8)
    # h_max = np.array((255, 255, 255), np.uint8)
    # brightness, contrast = 65 - 127, 170 - 127

    # CV transformations
    filtered = cv2.inRange(hsv, h_min, h_max)
    subtracted = cv2.subtract(filtered, gray)
    negative = cv2.bitwise_not(subtracted)
    res = apply_brightness_contrast(negative, brightness, contrast)

    return res