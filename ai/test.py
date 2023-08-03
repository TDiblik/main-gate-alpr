from ultralytics import YOLO

model = YOLO(f"./runs/detect/train2/weights/best.pt")
results = model.predict("./c.jpg")

result = results[0]
box = result.boxes[0]
print(box)