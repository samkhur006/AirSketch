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

while True:
    count = 0
    queue = deque([])
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:

            if hand_init_flag == False:
                hand_init_time = time.time()
                hand_init_flag = True

            # print(type(results.multi_hand_landmarks))
            x = [0] * 21
            y = [0] * 21
            enter=False
            for handLms in results.multi_hand_landmarks:

                #print(len(handLms))
                for i, lm in enumerate(handLms.landmark):
                    #print(i, lm)
                    enter=True
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x[i]=cx
                    y[i]=cy

                    if i == 8:
                        ct = time.time() - hand_init_time
                        print(cx, cy, ct)

                        if len(queue)==30:
                            print("This will be counted...")

                            index_points.append((cx, cy))
                            for index, item in enumerate(index_points):
                                if index == len(index_points) - 1:
                                    break
                                cv2.line(img, item, index_points[index + 1], [255, 0, 0], 2)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(queue) == 30:
                if queue[0]==True:
                    count=count-1
                queue.popleft()

            if enter==True and check(x,y)==True:
                queue.append(True)
                count=count+1
            else:
                queue.append(False)
            if count>=5:
                index_points=[]
                print("BREAKING")
                break

        cTime = time.time()
        fps   = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255))

        # print(i, img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

print("count is ",count)