import os
import cv2
from dotenv import load_dotenv

load_dotenv()

RTSP_CAPTURE_CONFIG = os.getenv("RTSP_CAPTURE_CONFIG") 
capture = cv2.VideoCapture(RTSP_CAPTURE_CONFIG)

while(capture.isOpened()):
    _, frame = capture.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()