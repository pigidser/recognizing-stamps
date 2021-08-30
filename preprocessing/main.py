import cv2
import numpy as np
import os
from utils import *


if __name__ == '__main__':
    def nothing(*arg):
        pass

cv2.namedWindow( "result" ) # создаем главное окно
cv2.namedWindow( "settings" ) # создаем окно настроек

# image_file = "51-page-8.png"
image_file = "20210812135619453-page-2.png"
image_folder = "content2\images"

file = os.path.join(os.getcwd(), image_folder, image_file)

if not os.path.isfile(file):
    raise Exception(f"File not found!")

img = cv2.imread(file)

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('h1', 'settings', 90, 255, nothing)
cv2.createTrackbar('s1', 'settings', 15, 255, nothing)
cv2.createTrackbar('v1', 'settings', 85, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
cv2.createTrackbar('brightness', 'settings', 5, 255, nothing)
cv2.createTrackbar('contrast', 'settings', 255, 255, nothing)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

while True:
    # flag, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
 
    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')
    brightness = cv2.getTrackbarPos('brightness', 'settings') - 127
    contrast = cv2.getTrackbarPos('contrast', 'settings') - 127

    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    filtered = cv2.inRange(hsv, h_min, h_max)

    subtracted = cv2.subtract(filtered, gray)

    negative = cv2.bitwise_not(subtracted)

    adjusted = apply_brightness_contrast(negative, brightness, contrast)

    cv2.imshow('result', np.hstack([filtered, subtracted, negative, adjusted]))
 
    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()