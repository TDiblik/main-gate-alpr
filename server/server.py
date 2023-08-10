import os
import sys
import cv2
import asyncio
import websockets
import numpy as np
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils

# Load env variables
load_dotenv()
DEBUG = os.getenv("DEBUG") == "True"
WS_PORT = int(os.getenv("WS_PORT"))
RTSP_CAPTURE_CONFIG = os.getenv("RTSP_CAPTURE_CONFIG") 
PURE_YOLO_MODEL_PATH = os.getenv("PURE_YOLO_MODEL_PATH") 
LICENSE_PLATE_YOLO_MODEL_PATH = os.getenv("LICENSE_PLATE_YOLO_MODEL_PATH") 

# Initialize global static variables
PURE_YOLO_MODEL = YOLO(PURE_YOLO_MODEL_PATH)
LICENSE_PLATE_YOLO_MODEL = YOLO(LICENSE_PLATE_YOLO_MODEL_PATH)
CAR_RELATED_LABELS = [
    utils.normalize_label('car'), 
    utils.normalize_label('motorcycle'), 
    utils.normalize_label('bus'), 
    utils.normalize_label('train'), 
    utils.normalize_label('truck')
]

CONNECTED_SOCKETS = []
async def handle_connection(websocket, path):
    global CONNECTED_SOCKETS
    CONNECTED_SOCKETS.append(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        CONNECTED_SOCKETS.remove(websocket)

def detect_images_from_capture(captured_frame: Image):
    number_of_yolo_boxes, yolo_boxes = utils.detect_with_yolo(PURE_YOLO_MODEL, captured_frame)
    if number_of_yolo_boxes == 0:
        print("No images of cars found")
        return

    utils.prepare_env_for_reading_license_plates(DEBUG)
    for (i, car_box) in enumerate(yolo_boxes):
        box_label = utils.normalize_label(
            PURE_YOLO_MODEL.names[int(car_box.cls)]
        )
        if box_label not in CAR_RELATED_LABELS:
            print(f"Found label \"{box_label}\", however it's not in CAR_RELATED_LABELS, skipping")
            continue

        x_min, y_min, x_max, y_max = car_box.xyxy.cpu().detach().numpy()[0]
        car_image = captured_frame.crop((x_min, y_min, x_max, y_max))
        if DEBUG:
            car_image.save(utils.gen_intermediate_file_name(f"cropped_car", "jpg", i))

async def detection_main_loop():
    while True:
        capture = cv2.VideoCapture(RTSP_CAPTURE_CONFIG)
        if capture.isOpened() is False:
            print("Unable to connect to video capture. Will try again after 5 seconds...")
            await asyncio.sleep(5)
            continue

        while(capture.isOpened()):
            able_to_read_frame, frame = capture.read()
            if able_to_read_frame is False:
                print("Could not read frame from video capture, reconnecting...")
                break
            if DEBUG:
                cv2.startWindowThread()
                cv2.namedWindow("frame")
                cv2.imshow("frame", cv2.resize(frame, (750, 750)))
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    os.exit()
            detect_images_from_capture(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)))
        capture.release()

async def main():
    ws_server = await websockets.serve(handle_connection, "localhost", WS_PORT)
    asyncio.create_task(detection_main_loop())
    await ws_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())