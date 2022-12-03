import cv2
import mediapipe as mp
import time
import sys

def draw_points(index_points):
    for index, item in enumerate(index_points): 
        if index == len(index_points) -1:
            break
        cv2.line(img, item, index_points[index + 1], [255, 0, 0], 2)

if __name__ == '__main__':
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


    while True:

        success, img = cap.read()
        img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:

            if hand_init_flag==False:
                hand_init_time=time.time()
                hand_init_flag = True

            # print(type(results.multi_hand_landmarks))
            for handLms in results.multi_hand_landmarks:
                # print(type(handLms))
                for i, lm in enumerate(handLms.landmark):
                    # print(id, lm)

                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if i==8:
                        ct = time.time()-hand_init_time
                        print(cx, cy, ct)

                        if ct >= hand_init_time_threshold:
                            print("This will be counted...")

                            index_points.append((cx, cy))
                            draw_points(index_points)
                            
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # print("index_points:", index_points)
            # print("strokes:", strokes)
        else:
            if len(strokes)!=0:
                print("A stroke ended...")
                new_stroke_flag = False
            
            draw_points(index_points)

        # cTime = time.time()
        # fps   = 1 / (cTime-pTime)
        # pTime = cTime
        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255))

        # print(i, img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

