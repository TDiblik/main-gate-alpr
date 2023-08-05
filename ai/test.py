import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO
import pytesseract

parser = argparse.ArgumentParser(
    prog="ALPR image manual tester",
    description="Manually nad visually view your AI's output",
)
parser.add_argument("trained_model", help="Path to your trained model")
parser.add_argument("img_to_test", help="Path to image to test")
args = parser.parse_args()

if __name__ == '__main__':
    print("Loading model")
    model = YOLO(args.trained_model)

    print("Starting prediction")
    result = model.predict(args.img_to_test)[0]
    if len(result.boxes) == 0:
        print("Didn't find any boxes/matches.")
        sys.exit(0)
    
    print("Plotting image")
    img = Image.open(args.img_to_test)
    plt.figure()
    plt.imshow(img)
    
    print("Plotting boxes")
    ax = plt.gca()
    for (i, box) in enumerate(result.boxes):
        x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]

        # Read using tesseract (https://wilsonmar.github.io/tesseract/)
        print(f'Reading box {i}')
        cropped_img_file_name = f'cropped_box_for_tesseract_{i}.jpg'
        cropped_img = img.crop((x_min, y_min, x_max, y_max))
        cropped_img.save(cropped_img_file_name, optimize=False)
        text_from_cropped_img = pytesseract.image_to_string(cropped_img_file_name, lang="eng", config="--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if not text_from_cropped_img:
            text_from_cropped_img = "no text found"
        text_from_cropped_img = text_from_cropped_img.strip()
        print(f'Cropped image read: {text_from_cropped_img}')

        # Plot
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="red")
        ax.add_patch(rect)

        # Add text above the rectangle
        plt.text(
            (x_min + x_max) / 2, y_min - 10, text_from_cropped_img, color="red", fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round"), 
            ha="center", va="baseline"
        )
    
    plt.show()