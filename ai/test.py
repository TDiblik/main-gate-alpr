import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils

parser = argparse.ArgumentParser(
    prog="ALPR image manual tester",
    description="Manually nad visually view your AI's output",
)
parser.add_argument("trained_model", help="Path to your trained model")
parser.add_argument("img_to_test", help="Path to image to test")
args = parser.parse_args()

BOOSTED_WIDTH = 500
if __name__ == '__main__':
    print("Loading model")
    license_plate_prediction_model = YOLO(args.trained_model)

    print("Starting prediction")
    img_to_test = Image.open(args.img_to_test)
    number_of_license_plate_boxes_found, license_plates_as_boxes = utils.detect_license_plates(license_plate_prediction_model, img_to_test)
    if number_of_license_plate_boxes_found == 0:
        print("Didn't find any boxes/matches.")
        sys.exit(0)
    
    print("Plotting image")
    plt.figure()
    plt.imshow(img_to_test)
    
    print("Plotting boxes")
    ax = plt.gca()
    utils.prepare_env_for_reading_license_plates(True)
    for (i, box) in enumerate(license_plates_as_boxes):
        license_plate_as_string = utils.read_license_plate(i, box, img_to_test, 500, 20, True)
        if len(license_plate_as_string) == 0:
            print(f"Result {i}, unable to find any characters of detected license plate")

        # Plot results
        x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="red")
        ax.add_patch(rect)
        plt.text(
            (x_min + x_max) / 2, y_min - 10, license_plate_as_string, color="red", fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round"), 
            ha="center", va="baseline"
        )

        # x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
        # print(f'Reading box {i}')
        # cropped_img_file_name = f'cropped_box_for_tesseract_{i}.jpg'
        # cropped_img = img_to_test.crop((x_min, y_min, x_max, y_max)).convert("L")
        # normal_width = int(x_max - x_min)
        # normal_height = int(y_max - y_min)
        # multiplier = BOOSTED_WIDTH / normal_width
        # cropped_img = img_to_test.crop((x_min, y_min, x_max, y_max)).convert("L").resize([int(normal_width * multiplier), int(normal_height * multiplier)])
        # cropped_img.save(cropped_img_file_name, optimize=False)

        # image = cv2.imread(cropped_img_file_name)
        # iwl_bb = utils.clean_plate_into_contours(image, BOOSTED_WIDTH)
        # iwl_wb = cv2.bitwise_not(iwl_bb)

        # cv2.imwrite(f"iwl_bb_{i}.jpg", iwl_bb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(f"iwl_wb_{i}.jpg", iwl_wb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # rectangles = utils.get_letter_rectangles_from_contours(iwl_wb)
        # for j, (x, y, w, h) in enumerate(rectangles):
        #     cropped_image = Image.open(f"iwl_wb_{i}.jpg")
        #     cropped_image = cropped_image.crop((x, y, x + w, y + h))

        #     white_spacing_each_side = 20
        #     original_width, original_height = cropped_image.size
        #     new_width = original_width + (white_spacing_each_side * 2)
        #     new_height = original_height + (white_spacing_each_side * 2)
        #     new_image = Image.new("RGB", (new_width, new_height), "white")
        #     new_image.paste(cropped_image, (white_spacing_each_side, white_spacing_each_side))
        #     new_image.save(f'block_{i}_{j}.jpg')

        #     char_from_img = pytesseract.image_to_string(f'block_{i}_{j}.jpg', lang="eng", config="--psm 13 --dpi 96 -c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789")
        #     char_from_img = str(char_from_img).strip()

        #     print(char_from_img)


        # text_from_cropped_img = pytesseract.image_to_string(cropped_img_file_name, lang="eng", config="--psm 7 --oem 1 --dpi 96 -c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789")
        # if not text_from_cropped_img:
        #     text_from_cropped_img = "no text found"
        # text_from_cropped_img = text_from_cropped_img.strip()
        # print(f'Cropped image read: {text_from_cropped_img}')
    
    plt.show()