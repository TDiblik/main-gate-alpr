import os
import sys
import uuid
import threading
import time
import cv2
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
SKIP_BEFORE_Y_MAX = float(os.getenv("SKIP_BEFORE_Y_MAX"))

# Initialize global static variables
PURE_YOLO_MODEL = YOLO(PURE_YOLO_MODEL_PATH)
LICENSE_PLATE_YOLO_MODEL = YOLO(LICENSE_PLATE_YOLO_MODEL_PATH)
CAR_RELATED_LABELS = [
    utils.normalize_label('car'), 
    utils.normalize_label('motorcycle'), 
    utils.normalize_label('bus'), 
    # utils.normalize_label('train'), 
    utils.normalize_label('truck'),
    utils.normalize_label('boat'), 
]

def _print(string):
    if DEBUG: print(string)

############ Web socket server ############
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
def run_websocket_server():
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websockets.serve(handle_connection, "localhost", WS_PORT))
    loop.run_forever()

############ Video capture ############
LATEST_FRAME = None
def run_video_capture():
    global LATEST_FRAME
    while True:
        capture = cv2.VideoCapture(RTSP_CAPTURE_CONFIG)
        if capture.isOpened() is False:
            _print("Unable to connect to video capture. Will try again after 5 seconds...")
            LATEST_FRAME = None
            time.sleep(5)
            continue

        while(capture.isOpened()):
            able_to_read_frame, frame = capture.read()
            if able_to_read_frame is False:
                _print("Could not read frame from video capture, reconnecting...")
                LATEST_FRAME = None
                break

            if DEBUG:
                cv2.startWindowThread()
                cv2.namedWindow("frame")
                cv2.imshow("frame", cv2.resize(frame, (750, 750)))
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    os.exit()

            LATEST_FRAME = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8))

        capture.release()

############ Detection ############
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
        if y_max < SKIP_BEFORE_Y_MAX:
            _print(f"Found car, however it's too far \"{y_max}\" (req \"{SKIP_BEFORE_Y_MAX}\"), skipping")
            continue

        car_image = captured_frame.crop((x_min, y_min, x_max, y_max))
        if DEBUG:
            car_image.save(utils.gen_intermediate_file_name(f"cropped_car", "jpg", i))

        number_of_license_plate_boxes_found, license_plates_as_boxes = utils.detect_with_yolo(LICENSE_PLATE_YOLO_MODEL, car_image)
        if number_of_license_plate_boxes_found == 0:
            continue

        for (j, license_plate_box) in enumerate(license_plates_as_boxes):
            license_plate_image, license_plate_as_string = utils.read_license_plate(f"{i}_{j}", license_plate_box, car_image, 500, 20, DEBUG, MINIMUM_NUMBER_OF_CHARS_FOR_MATCH)
            if license_plate_as_string == "":
                _print(f"Car {i} ; Result {j}, unable to find any characters of detected license plate")
                continue
            if len(license_plate_as_string) < MINIMUM_NUMBER_OF_CHARS_FOR_MATCH:
                _print(f"Found license plate {license_plate_as_string}, but it's shorter than {MINIMUM_NUMBER_OF_CHARS_FOR_MATCH}")
                continue

            _print(y_max)
            _print(f"Found license plate {license_plate_as_string}")
            license_plates_recognized.append((car_image, license_plate_image, license_plate_as_string))

    return license_plates_recognized

# any => Image (it cannot be used as type)
def validate_results_between_rounds(recognitions_between_rounds: list[list[(any, any, str)]], number_of_occurrences_to_be_valid: int):
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

async def detection():
    recognitions_between_rounds = []
    while True:
        if LATEST_FRAME is None:
            _print("LATEST_FRAME is None, nothing to do, sleeping for 1 second...")
            time.sleep(1)
            continue

        license_plates_recognized = detect_license_plates_from_frame(LATEST_FRAME.copy())
        if len(license_plates_recognized) == 0:
            continue

        recognitions_between_rounds.append(license_plates_recognized)
        if len(recognitions_between_rounds) != NUMBER_OF_VALIDATION_ROUNDS:
            continue
        
        validated_results = validate_results_between_rounds(recognitions_between_rounds, NUMBER_OF_OCCURRENCES_TO_BE_VALID)
        if len(validated_results) == 0:
            if recognitions_between_rounds != []:
                recognitions_between_rounds.pop(0)
            continue
            
        _print("Sending results: ")
        _print(validated_results)
        for res in validated_results:
            car_image = utils.img_to_bytes(res[0])
            license_plate_image = utils.img_to_bytes(res[1])
            license_plate_as_string = str(res[2]) # just to make sure it's string
            license_plate_as_string = license_plate_as_string[:3] + " " + license_plate_as_string[3:]
            license_plate_uuid = str(uuid.uuid4())
            license_plate_formated_string = license_plate_as_string + " => " + license_plate_uuid
            for socket in CONNECTED_SOCKETS:
                # try:
                await socket.send(car_image)
                await socket.send(license_plate_image)
                await socket.send(license_plate_formated_string)
                # except Exception as e:
                #     _print(e)

        recognitions_between_rounds = []

def run_detection():
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(detection())

if __name__ == "__main__":
    os.environ['OMP_THREAD_LIMIT'] = '1'

    websocket_thread = threading.Thread(target=run_websocket_server)
    capture_thread = threading.Thread(target=run_video_capture)
    detection_thread = threading.Thread(target=run_detection)

    websocket_thread.start()
    capture_thread.start()
    detection_thread.start()

    websocket_thread.join()
    capture_thread.join()
    detection_thread.join()