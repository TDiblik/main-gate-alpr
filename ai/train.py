import os
import torch
from ultralytics import YOLO, settings

MODEL_PATH = "yolov8n.pt" # nano model
# MODEL_PATH = "yolov8s.pt" # small model
# MODEL_PATH = "yolov8m.pt" # medium model
# MODEL_PATH = "yolov8l.pt" # large model
# MODEL_PATH = "yolov8x.pt" # huge model

# https://github.com/ultralytics/ultralytics/issues/348
if __name__ == "__main__":
    torch.cuda.empty_cache()

    # https://docs.ultralytics.com/quickstart/?h=settings#modifying-settings
    settings.update({"datasets_dir": os.path.join(os.getcwd(), "./training_data_preprocessed/")})
    settings.save()

    # Load model
    model = YOLO(MODEL_PATH)

    # https://docs.ultralytics.com/modes/train/#arguments
    # Specifically pay attention to `batch`, if you encouter the `torch.cuda.OutOfMemoryError: CUDA out of memory.`, just make the batches smaller.
    model.train(data="./data.yaml", epochs=100, patience=50, batch=16)