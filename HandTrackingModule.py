import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectCon = detectCon
        self.trackCon = trackCon

        # Hand module from "mp"
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):                
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:              
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
        return lmList

def main():
    # Frame rate
    pTime = 0
    cTime = 0
    
    # Activate webcam
    cap = cv.VideoCapture(0)
    
    detector = HandDetector()

    while True:
        isTrue, frame = cap.read()
        
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, 0, draw=False)

        # print position 4 (thumb tip)
        if len(lmList) != 0:
            print(lmList[4])
            cv.circle(frame, (lmList[4][1], lmList[4][2]), 10, (255, 0, 255), cv.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        fps_display = "FPS: " + str(int(fps))

        cv.putText(frame, fps_display, (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv.imshow("Video", frame)
        if cv.waitKey(1) & 0xFF == ord('x'):
            break

if __name__ == "__main__":
    main()