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
MINIMUM_NUMBER_OF_CHARS_FOR_MATCH = int(os.getenv("MINIMUM_NUMBER_OF_CHARS_FOR_MATCH"))
NUMBER_OF_VALIDATION_ROUNDS = int(os.getenv("NUMBER_OF_VALIDATION_ROUNDS"))
NUMBER_OF_OCCURRENCES_TO_BE_VALID = int(os.getenv("NUMBER_OF_OCCURRENCES_TO_BE_VALID"))

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

def _print(string):
    if DEBUG: print(string)

CONNECTED_SOCKETS = []
async def handle_connection(websocket, path):
    global CONNECTED_SOCKETS
    CONNECTED_SOCKETS.append(websocket)
    try:
        await websocket.send("echo")
        async for message in websocket:
            pass
    finally:
        CONNECTED_SOCKETS.remove(websocket)

# [(Image, Image, str)] => array of (car image, license plate image, license plate as string)
def detect_license_plates_from_frame(captured_frame: Image) -> [(Image, Image, str)]:
    number_of_yolo_boxes, yolo_boxes = utils.detect_with_yolo(PURE_YOLO_MODEL, captured_frame)
    if number_of_yolo_boxes == 0:
        _print("No images of cars found")
        return []

    license_plates_recognized = []
    utils.prepare_env_for_reading_license_plates(DEBUG)
    for (i, car_box) in enumerate(yolo_boxes):
        box_label = utils.normalize_label(
            PURE_YOLO_MODEL.names[int(car_box.cls)]
        )
        if box_label not in CAR_RELATED_LABELS:
            _print(f"Found label \"{box_label}\", however it's not in CAR_RELATED_LABELS, skipping")
            continue

        x_min, y_min, x_max, y_max = car_box.xyxy.cpu().detach().numpy()[0]
        car_image = captured_frame.crop((x_min, y_min, x_max, y_max))
        if DEBUG:
            car_image.save(utils.gen_intermediate_file_name(f"cropped_car", "jpg", i))

        number_of_license_plate_boxes_found, license_plates_as_boxes = utils.detect_with_yolo(LICENSE_PLATE_YOLO_MODEL, car_image)
        if number_of_license_plate_boxes_found == 0:
            continue

        for (j, license_plate_box) in enumerate(license_plates_as_boxes):
            license_plate_image, license_plate_as_string = utils.read_license_plate(f"{i}_{j}", license_plate_box, car_image, 500, 20, True, MINIMUM_NUMBER_OF_CHARS_FOR_MATCH)
            if license_plate_as_string == "":
                _print(f"Car {i} ; Result {j}, unable to find any characters of detected license plate")
                continue
            if len(license_plate_as_string) < MINIMUM_NUMBER_OF_CHARS_FOR_MATCH:
                _print(f"Found license plate {license_plate_as_string}, but it's shorter than {MINIMUM_NUMBER_OF_CHARS_FOR_MATCH}")
                continue

            _print(f"Found license plate {license_plate_as_string}")
            license_plates_recognized.append((car_image, license_plate_image, license_plate_as_string))

    return license_plates_recognized

# any => Image (but it cannot be used as type)
def validate_results(recognitions_between_rounds: list[list[(any, any, str)]], number_of_occurrences_to_be_valid: int):
    license_plate_counts = {}
    for recognitions in recognitions_between_rounds:
        for _, _, license_plate_as_string in recognitions:
            if license_plate_as_string in license_plate_counts:
                license_plate_counts[license_plate_as_string] += 1
            else:
                license_plate_counts[license_plate_as_string] = 1
    
    validated_recognitions = []
    for license_plate, times_found in license_plate_counts.items():
        if times_found < number_of_occurrences_to_be_valid:
            continue
        
        should_break_recognitions_loop = False
        for recognitions in recognitions_between_rounds:
            if should_break_recognitions_loop: break
            for recognized_car_image, recognized_license_plate_image, recognized_license_plate_as_string in recognitions:
                if should_break_recognitions_loop: break
                if license_plate == recognized_license_plate_as_string:
                    validated_recognitions.append((recognized_car_image, recognized_license_plate_image, recognized_license_plate_as_string))
                    should_break_recognitions_loop = True

    return validated_recognitions

async def detection_main_loop():
    while True:
        capture = cv2.VideoCapture(RTSP_CAPTURE_CONFIG)
        if capture.isOpened() is False:
            _print("Unable to connect to video capture. Will try again after 5 seconds...")
            await asyncio.sleep(5)
            continue

        recognitions_between_rounds = []
        while(capture.isOpened()):
            await asyncio.sleep(0.01) # Checkup on other ascio tasks
            able_to_read_frame, frame = capture.read()
            if able_to_read_frame is False:
                _print("Could not read frame from video capture, reconnecting...")
                break

            if DEBUG:
                cv2.startWindowThread()
                cv2.namedWindow("frame")
                cv2.imshow("frame", cv2.resize(frame, (750, 750)))
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    os.exit()

            license_plates_recognized = detect_license_plates_from_frame(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)))
            if len(license_plates_recognized) == 0:
                continue

            recognitions_between_rounds.append(license_plates_recognized)
            if len(recognitions_between_rounds) != NUMBER_OF_VALIDATION_ROUNDS:
                continue
            
            validated_results = validate_results(recognitions_between_rounds, NUMBER_OF_OCCURRENCES_TO_BE_VALID)
            if len(validated_results) == 0:
                recognitions_between_rounds = []
                continue
                
            _print("Sending result: ")
            _print(validated_results)
            for socket in CONNECTED_SOCKETS:
                for res in validated_results:
                    await socket.send(res[2])
                # await socket.send(validated_results)

            recognitions_between_rounds = []

        capture.release()

async def main():
    ws_server = await websockets.serve(handle_connection, "localhost", WS_PORT)
    asyncio.create_task(detection_main_loop())
    await ws_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())