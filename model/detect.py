from ultralytics import YOLO
import torch
import numpy as np
import cv2
import base64
from PIL import Image
import io

# Path to your trained model
MODEL_PATH = "model/best.pt"

model = YOLO("model/best.pt")
model.model.fuse = lambda *a, **k: model.model  # Monkey patch to skip fusing
def run_yolo_inference_on_numpy(img_rgb):
    """
    img_rgb: numpy array (H,W,3) in RGB format
    returns: list of detections [ {xmin, ymin, xmax, ymax, conf, class_name}... ]
    """
    # ultralytics accepts numpy array in BGR or PIL; convert to BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(source=img_bgr, imgsz=640, conf=0.25, iou=0.45, verbose=False)
    detections = []
    # results is list with a single element for this image
    r = results[0]
    boxes = r.boxes
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        conf = float(box.conf.cpu().numpy())
        cls = int(box.cls.cpu().numpy())
        name = model.names.get(cls, str(cls))
        detections.append({
            "xmin": int(xyxy[0]), "ymin": int(xyxy[1]),
            "xmax": int(xyxy[2]), "ymax": int(xyxy[3]),
            "conf": conf, "class": name
        })
    return detections
