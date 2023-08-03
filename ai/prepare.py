import os
import zipfile
import cv2
import shutil
import numpy as np
import xml.etree.ElementTree as ET

def convert_pascal_voc_into_yolo_annotation(annotation_path: str):
    tree = ET.parse(annotation_path)
    root_elem = tree.getroot()

    size_elem = root_elem.find("size")
    image_width = int(size_elem.find("width").text)
    image_height = int(size_elem.find("height").text)

    yolo_annotation = ''
    objects = root_elem.findall("object")
    for obj in objects:
        name = str(obj.find("name").text)
        if name not in ["license-plate", "license", "licence", "num_plate", "number_plate"]:
            print(f'Skipping objects: {name}')
            continue

        bounding_box_elem = obj.find("bndbox")

        x_min = int(bounding_box_elem.find("xmin").text)
        x_max = int(bounding_box_elem.find("xmax").text)
        y_min = int(bounding_box_elem.find("ymin").text)
        y_max = int(bounding_box_elem.find("ymax").text)

        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        yolo_width = (x_max - x_min) / image_width
        yolo_height = (y_max - y_min) / image_height

        yolo_annotation += f'0 {x_center} {y_center} {yolo_width} {yolo_height}\n'

    return yolo_annotation.rstrip()

def train_validation_split(input_data: list[any]):
    total_size = len(input_data)
    train_size = int(0.8 * total_size)
    
    np.random.shuffle(input_data)
    
    train_data = input_data[:train_size]
    validation_data = input_data[train_size:]
    return train_data, validation_data

def save_dataset_item(image_path: str, yolo_annotation: str, image_destination_path: str, label_destination_path: str, data_item_name: str):
    new_image_path = os.path.join(image_destination_path, f'{data_item_name}.jpg')
    if image_path.endswith(".png"):
        image = cv2.imread(image_path)
        cv2.imwrite(new_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        shutil.copy(image_path, new_image_path)
    else:
        print("Unable to work with formats outside of .jpg and .png - Implment them yourself and submit a PR :D.")
        return
    
    new_yolo_annotation_path = os.path.join(label_destination_path, f'{data_item_name}.txt')
    yolo_annotation_file = open(new_yolo_annotation_path, "w")
    yolo_annotation_file.write(yolo_annotation)
    yolo_annotation_file.close()

RESOURCES_DIRECTORY = "./resources/"

INPUT_DATASET_ROOT = "./training_data/"
ANDREWMVD_DATASET_ROOT = os.path.join(INPUT_DATASET_ROOT, "andrewmvd_dataset/")
ASLANAHMEDOV_DATASET_ROOT = os.path.join(INPUT_DATASET_ROOT, "aslanahmedov_dataset/")

OUTPUT_DIRECTORY = "./training_data_preprocessed/"
OUTPUT_TRAINING_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "training/")
OUTPUT_TRAINING_IMAGES_DIRECTORY = os.path.join(OUTPUT_TRAINING_DIRECTORY, "images/")
OUTPUT_TRAINING_LABELS_DIRECTORY = os.path.join(OUTPUT_TRAINING_DIRECTORY, "labels/")
OUTPUT_VALIDATION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "validation/")
OUTPUT_VALIDATION_IMAGES_DIRECTORY = os.path.join(OUTPUT_VALIDATION_DIRECTORY, "images/")
OUTPUT_VALIDATION_LABELS_DIRECTORY = os.path.join(OUTPUT_VALIDATION_DIRECTORY, "labels/")


if __name__ == '__main__':
    print("Preparing environment")

    if os.path.exists(INPUT_DATASET_ROOT):
        shutil.rmtree(INPUT_DATASET_ROOT)
    os.mkdir(INPUT_DATASET_ROOT)
    os.mkdir(ANDREWMVD_DATASET_ROOT)
    os.mkdir(ASLANAHMEDOV_DATASET_ROOT)

    if os.path.exists(OUTPUT_DIRECTORY):
        shutil.rmtree(OUTPUT_DIRECTORY)
    os.mkdir(OUTPUT_DIRECTORY)
    os.mkdir(OUTPUT_TRAINING_DIRECTORY)
    os.mkdir(OUTPUT_TRAINING_IMAGES_DIRECTORY)
    os.mkdir(OUTPUT_TRAINING_LABELS_DIRECTORY)
    os.mkdir(OUTPUT_VALIDATION_DIRECTORY)
    os.mkdir(OUTPUT_VALIDATION_IMAGES_DIRECTORY)
    os.mkdir(OUTPUT_VALIDATION_LABELS_DIRECTORY)

    print("Loading and parsing andrewmvd dataset")
    with zipfile.ZipFile(os.path.join(RESOURCES_DIRECTORY, "andrewmvd_dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(ANDREWMVD_DATASET_ROOT)
    andrewmvd_dataset = []
    andrewmvd_images_path = os.path.join(ANDREWMVD_DATASET_ROOT, "images")
    for image_file_name in os.listdir(andrewmvd_images_path):
        image_path = os.path.join(andrewmvd_images_path, image_file_name)
        if os.path.isfile(image_path) is False:
            continue

        annotation_path = os.path.join(ANDREWMVD_DATASET_ROOT, "annotations", f'{image_file_name.removesuffix(".png")}.xml')
        if os.path.isfile(annotation_path) is False:
            print(f'Annotation {annotation_path} missing.')
            continue

        yolo_annotation = convert_pascal_voc_into_yolo_annotation(annotation_path)
        andrewmvd_dataset.append((image_path, yolo_annotation))

    print("Loading and parsing aslanahmedov dataset")
    with zipfile.ZipFile(os.path.join(RESOURCES_DIRECTORY, "aslanahmedov_dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(ASLANAHMEDOV_DATASET_ROOT)
    aslanahmedov_dataset = []
    aslanahmedov_images_path = os.path.join(ASLANAHMEDOV_DATASET_ROOT, "images")
    for image_file_name in os.listdir(aslanahmedov_images_path):
        image_path = os.path.join(aslanahmedov_images_path, image_file_name)
        if os.path.isfile(image_path) is False or image_path.endswith(".jpeg") is False:
            continue

        annotation_path = os.path.join(os.path.dirname(image_path), f'{image_file_name.removesuffix(".jpeg")}.xml')
        if os.path.isfile(annotation_path) is False:
            print(f'Annotation {annotation_path} missing.')
            continue

        yolo_annotation = convert_pascal_voc_into_yolo_annotation(annotation_path)
        aslanahmedov_dataset.append((image_path, yolo_annotation))

    print("Spliting all datasets into training and validation data")
    andrewmvd_training_dataset, andrewmvd_validation_dataset = train_validation_split(andrewmvd_dataset)
    aslanahmedov_training_dataset, aslanahmedov_validation_dataset = train_validation_split(aslanahmedov_dataset)

    print("Merging all training and validation datasets")
    training_dataset = andrewmvd_training_dataset + aslanahmedov_training_dataset
    validation_dataset = andrewmvd_validation_dataset + aslanahmedov_validation_dataset

    print("Shuffling finalized datasets")
    np.random.shuffle(training_dataset)
    np.random.shuffle(validation_dataset)

    print("Saving training data")
    for i, (image_path, yolo_annotation) in enumerate(training_dataset):
        save_dataset_item(image_path, yolo_annotation, OUTPUT_TRAINING_IMAGES_DIRECTORY, OUTPUT_TRAINING_LABELS_DIRECTORY, i)

    print("Saving validation data")
    for i, (image_path, yolo_annotation) in enumerate(validation_dataset):
        save_dataset_item(image_path, yolo_annotation, OUTPUT_VALIDATION_IMAGES_DIRECTORY, OUTPUT_VALIDATION_LABELS_DIRECTORY, i)