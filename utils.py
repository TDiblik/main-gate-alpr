import os
import shutil
import cv2
import imutils
from skimage.filters import threshold_local
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract

# Returns number of results + results as boxes
def detect_with_yolo(preloaded_model: YOLO, car_image: Image) -> (int, any):
    result = preloaded_model.predict(car_image)[0]
    return (len(result.boxes), result.boxes)

def normalize_label(label):
    return label.strip().lower()

def clean_plate_into_contours(cvImage: np.ndarray, fixed_width: int) -> np.ndarray:
    plate_img = cvImage.copy()
    
    # Set non-black/white pixels to white
    # gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # _, binary_img = cv2.threshold(gray_plate_img, 128, 255, cv2.THRESH_BINARY)
    # non_black_white_mask = binary_img == 0
    # plate_img[non_black_white_mask] = [255, 255, 255]  

    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method='gaussian')
    thresh = (V > T).astype('uint8') * 255
    thresh = cv2.bitwise_not(thresh)
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    return thresh

def get_letter_rectangles_from_contours(iwl):
    contours,_ = cv2.findContours(iwl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rectangles=[]
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        if(h < (iwl.shape[0] / 5) or w > (iwl.shape[1] / 5)):
            continue
        rectangles.append([x,y,w,h])
    final_rect = []
    for (x, y, w, h) in rectangles:
        flag = True
        for(x2, y2, w2, h2) in rectangles:
            if x > x2 and y > y2 and x + w < x2 + w2 and y + h < y2 + h2:
                flag = False
                break
        if flag:
            final_rect.append([x,y,w,h])
    rectangles = final_rect
    rectangles.sort()
    return rectangles

def gen_intermediate_file_name(filename: str, file_type: str, iter_number: int | str):
    return f"./intermediate_detection_files/{filename}_{iter_number}.{file_type}"

# This function should be called right after getting license plate boxes
def prepare_env_for_reading_license_plates(should_save_intermediate_files: bool):
    if should_save_intermediate_files:
        if os.path.exists("./intermediate_detection_files/"):
            shutil.rmtree("./intermediate_detection_files/")
        os.mkdir("./intermediate_detection_files/")

# Read single license plate box
# Returns license plate as string
def read_license_plate(iter_number: int, box: any, original_image: Image, width_boost: int, additional_white_spacing_each_side: int, debug: bool) -> str:
    # Crop image
    x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
    original_width = x_max - x_min
    original_height = y_max - y_min
    boost_multiplier = width_boost / original_width
    boosted_width = int(original_width * boost_multiplier)
    boosted_height = int(original_height * boost_multiplier)
    license_plate_cropped_img = original_image.crop(  # crop to license plate
        (x_min, y_min, x_max, y_max)
    ).convert( # black and white images make preprocessing more effective
        "L"
    ).resize( # resizing makes recognition more effective
        [boosted_width, boosted_height]
    ).crop( # crop from left and right, because license plate recognition matches with overflow
        ((55, 0, boosted_width - 20, boosted_height)) 
    )
    if debug:
        license_plate_cropped_img.save(gen_intermediate_file_name("cropped_license_plate_full", "jpg", iter_number))
        
    # Pre-process the image
    license_plate_cropped_img_as_np_array = cv2.cvtColor(np.array(license_plate_cropped_img), cv2.COLOR_GRAY2BGR)
    iwl_bb = clean_plate_into_contours(license_plate_cropped_img_as_np_array, width_boost)
    iwl_wb = cv2.bitwise_not(iwl_bb)
    if debug:
        cv2.imwrite(gen_intermediate_file_name("iwl_bb", "jpg", iter_number), iwl_bb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(gen_intermediate_file_name("iwl_wb", "jpg", iter_number), iwl_wb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
    # Get each letter
    resulting_license_plate_string = ""
    letter_rectangles = get_letter_rectangles_from_contours(iwl_wb)
    iwl_wb_pil = Image.fromarray(iwl_wb, mode="L")
    
    for j, (x, y, w, h) in enumerate(letter_rectangles):
        # letter_box_cropped_img = license_plate_cropped_img.crop((x, y, x + w, y + h))
        letter_box_cropped_img = iwl_wb_pil.crop((x, y, x + w, y + h))
        new_letter_box_img = Image.new(
            "L", (
                letter_box_cropped_img.width + (additional_white_spacing_each_side * 2), 
                letter_box_cropped_img.height + (additional_white_spacing_each_side * 2)
            ), "white"
        )
        new_letter_box_img.paste(letter_box_cropped_img, (additional_white_spacing_each_side, additional_white_spacing_each_side))
        if debug:
            new_letter_box_img.save(gen_intermediate_file_name("cropped_image", "jpg", f"{iter_number}_{j}"))

        # Read using tesseract (https://wilsonmar.github.io/tesseract/)
        char_from_img = pytesseract.image_to_string(new_letter_box_img, lang="eng", config="--psm 13 --dpi 96 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        char_from_img = str(char_from_img).replace("O", "0").strip()
        if debug:
            print(char_from_img)
        resulting_license_plate_string += char_from_img
        
    return resulting_license_plate_string