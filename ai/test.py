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

if __name__ == '__main__':
    print("Loading model")
    license_plate_prediction_model = YOLO(args.trained_model)

    print("Starting prediction")
    img_to_test = Image.open(args.img_to_test)
    number_of_license_plate_boxes_found, license_plates_as_boxes = utils.detect_with_yolo(license_plate_prediction_model, img_to_test, True)
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
        _, license_plate_as_string = utils.read_license_plate(i, box, img_to_test, 500, 20, True, False, 4) # defaults from .env.development
        if len(license_plate_as_string) == 0:
            print(f"Result {i}, unable to find any characters of detected license plate")
        print(f"Found license plate: {license_plate_as_string}")

        # Plot results
        x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="red")
        ax.add_patch(rect)
        plt.text(
            (x_min + x_max) / 2, y_min - 10, license_plate_as_string, color="red", fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round"), 
            ha="center", va="baseline"
        )
    
    plt.show()