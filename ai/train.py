import os
from ultralytics import YOLO, settings

# MODEL_PATH = "./resources/yolov8n.pt" # nano model
# MODEL_PATH = "./resources/yolov8s.pt" # small model
# MODEL_PATH = "./resources/yolov8m.pt" # medium model
# MODEL_PATH = "./resources/yolov8l.pt" # large model
MODEL_PATH = "./resources/yolov8x.pt" # huge model

model = YOLO(MODEL_PATH)

# https://docs.ultralytics.com/quickstart/?h=settings#modifying-settings
settings.update({"datasets_dir": os.path.join(os.getcwd(), "./training_data_preprocessed/")})

# https://docs.ultralytics.com/modes/train/#arguments
model.train(data="./data.yaml", epochs=100, patience=50, batch=-1)