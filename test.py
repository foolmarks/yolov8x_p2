from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image

#model_path = hf_hub_download("soyeollee/yolov8x-p2-coco", "model.pt")
model_path = "model.onnx"

model = YOLO(model_path)

image_path = "tst.jpg"
results = model(Image.open(image_path))


# Inspect results
for r in results:
    print(r.boxes)
    print(r.masks if hasattr(r, "masks") else None)
