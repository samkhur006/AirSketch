import cv2
import mediapipe as mp
import time
import sys
from bisect import *
from config import *

def draw_points(index_points):
    for index, item in enumerate(index_points): 
        if index == len(index_points) -1:
            break
        cv2.line(img, item, index_points[index + 1], [255, 0, 0], 2)

def createStrokes(index_points, cx, cy, ct):
    if len(index_points)>init_index_points_count:
        """
        Checks only after 15 index points are captured (Approx for a sec)
        Tries to find the time which was almost 1 sec prior to the current time
        If such a time is found, then finds the average of all the x and y coordinates during
        the last second and checks if the hand was actually stationary. 
        If the hand has not moved in the last second, then a new stroke is created
        """
        lb = bisect_left(timestamps, ct-1, lo=max(0,len(timestamps)-init_index_points_count), hi=len(timestamps))
        if ct - timestamps[lb] >= new_stroke_time_threshold:            
            avg_x = 0
            avg_y = 0
            for index_point in index_points[lb:]:
                avg_x += index_point[0]
                avg_y += index_point[1]
            avg_x /= len(index_points[lb:])
            avg_y /= len(index_points[lb:])
            # print(avg_x, avg_y)

            if(abs(avg_x - cx) < 2.0 or abs(avg_y - cy) < 2.0):
                ### Create new stroke
                return True
    return False

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    init_time = time.time()

    index_points = []
    timestamps = []
    strokes = dict()

    stroke_count = 0
    hand_init_flag = False

    hand_init_time = float()

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

            for handLms in results.multi_hand_landmarks:
                for i, lm in enumerate(handLms.landmark):
                    if i==8:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        ct = time.time()-hand_init_time
                        print(cx, cy, ct)

                        if ct >= hand_init_time_threshold:
                            index_points.append((cx, cy))
                            timestamps.append(ct)

                            if createStrokes(index_points, cx, cy, ct):
                                stroke_count += 1
                                print("stroke_count:", stroke_count)
                                if stroke_count>1 and len(strokes[stroke_count-1])==1:
                                    stroke_count -= 1
                                    print("Removed the last stroke as the last stroke had only one points")
                                    print("stroke_count:", stroke_count)
                                strokes[stroke_count] = []
                            
                            if stroke_count>0:
                                strokes[stroke_count].append((cx, cy, ct))
                                print(strokes)
                                print("-"*20)
                                draw_points(index_points)
                            
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # print("index_points:", index_points)
            # print("strokes:", strokes)
        else:
            if len(strokes)!=0:
                print("Hand is out of drawing area!!!")
                new_stroke_flag = False
            
            draw_points(index_points)

        # cTime = time.time()
        # fps   = 1 / (cTime-pTime)
        # pTime = cTime
        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255))

        # print(i, img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

