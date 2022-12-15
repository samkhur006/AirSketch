import cv2
import mediapipe as mp
import time
import sys
import numpy as np

from collections import deque


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

init_time = time.time()

index_points = []

strokes = dict()
hand_init_flag = False

hand_init_time = float()
hand_init_time_threshold = 1.0

new_stroke_time_threshold = 0.5

def check(x,y):
    flag=True
    xx=np.array([x[6]-x[5],x[7]-x[5],x[8]-x[5]])
    yy=np.array([y[6]-y[5],y[7]-y[5],y[8]-y[5]])
    arr=np.arctan2(xx,yy)
    if(abs(arr[1]-arr[0])>0.17 or abs(arr[2]-arr[0])>0.17):
        return False
    xx = np.array([x[10] - x[9], x[11] - x[9], x[12] - x[9]])
    yy = np.array([y[10] - y[9], y[11] - y[9], y[12] - y[9]])
    arr = np.arctan2(xx, yy)
    if (abs(arr[1] - arr[0]) > 0.17 or abs(arr[2] - arr[0]) > 0.17):
        return False
    xx = np.array([x[14] - x[13], x[15] - x[13], x[16] - x[13]])
    yy = np.array([y[14] - y[13], y[15] - y[13], y[16] - y[13]])
    arr = np.arctan2(xx, yy)
    if (abs(arr[1] - arr[0]) > 0.21 or abs(arr[2] - arr[0]) > 0.21):
        return False
    xx = np.array([x[18] - x[17], x[19] - x[17], x[20] - x[17]])
    yy = np.array([y[18] - y[17], y[19] - y[17], y[20] - y[17]])
    arr = np.arctan2(xx, yy)
    if (abs(arr[1] - arr[0]) > 0.17 or abs(arr[2] - arr[0]) > 0.17):
        return False
    xx = np.array([x[4] - x[3], x[3] - x[2], x[2] - x[1]])
    yy = np.array([y[4] - y[3], y[3] - y[2], y[2] - y[1]])
    arr = np.arctan2(xx, yy)
    if (abs(arr[2] - arr[0]) > 0.21):
        return False
    return True