import cv2 as cv
from cv2 import FILLED
import mediapipe as mp
import time
import os.path as path
import numpy as np
import math

# Module
import HandTrackingModule as HTM

# Libary for Pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

######
# Change volume stuff
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
#####

#######
wCam, hCam = 640, 360
#######

# img rate
pTime = 0
cTime = 0
  
# Activate webcam
cap = cv.VideoCapture(0)

# Set cam size
cap.set(3, wCam)
cap.set(4, hCam)

"""
# Path to videos folder
dir_path = path.dirname(path.realpath(__file__))
videos_folder_path = path.join(dir_path, "HandVideos/")

# Open up the video selected
video_path = path.join(videos_folder_path, "test_hands.mp4")
cap = cv.VideoCapture(video_path)
"""

detector = HTM.HandDetector(detectCon=0.7)

vol = 0
volBar = 300
volPer = 0

while True:
    isTrue, img = cap.read()

    """
    # resize image
    height, width, layers = img.shape
    img = cv.resize(img, (wCam, hCam))
    """

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get position 4 and position 8
        id_1 = 4
        id_2 = 8

        x1, y1 = lmList[id_1][1], lmList[id_1][2]
        x2, y2 = lmList[id_2][1], lmList[id_2][2]
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2     # Center point

        # Draw 3 point
        cv.circle(img, (x1, y1), 10, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 10, (255, 0, 255), cv.FILLED)
        cv.circle(img, (xc, yc), 10, (255, 0, 255), cv.FILLED)
        
        # Create line betwwen 2 postion
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        length = math.hypot(x2 - x1, y2 - y1)
        
        # Hand Range: 20 - 150
        # Volume Range: -96 - 0
        vol = np.interp(length, [20, 150], [minVol, maxVol])
        volBar = np.interp(length, [20, 150], [300, 75])    # Position to draw volume bar
        # Volume Percentage: 0 - 100
        volPer = np.interp(length, [20, 150], [0, 100])

        # Show Variables
        print(int(length), vol)

        # Set volume
        volume.SetMasterVolumeLevel(vol, None)

        # "Click" action
        if length <= 20:
            cv.circle(img, (xc, yc), 10, (0, 255, 0), cv.FILLED)

    # Display volume bar + volume percentage
    cv.rectangle(img, (50, 75), (85, 300), (255, 0, 0), 3, 3)
    cv.rectangle(img, (50, int(volBar)), (85, 300), (255, 0, 0), cv.FILLED)
    cv.putText(img, f'{int(volPer)} %', (30, 350), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # Calculate + display FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    fps_display = "FPS: " + str(int(fps))

    # Display FPS counter
    cv.putText(img, fps_display, (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Display image
    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord('x'):
        break
