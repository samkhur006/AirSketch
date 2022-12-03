import cv2
import mediapipe as mp
import time
import sys
import numpy as np

a = [(375, 193), (364, 113), (277, 20), (271, 16), (52, 106), (133, 266), (289, 296), (372, 282)]

image = np.zeros([512, 512, 3],np.uint8)

cv2.polylines(image, a, isClosed = False, color = (0,255,0), thickness = 3)