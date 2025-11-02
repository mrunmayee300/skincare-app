from ultralytics import YOLO

# Use the yolov8n (nano) model for quick experiments
model = YOLO('yolov8n.pt')  # or yolov8s.pt / custom
# dataset YAML should point to paths: train, val, names
# Example command if you have 'data.yaml'

model.train(data="acne_data/data.yaml", epochs=3, imgsz=640, batch=16, name="acne_experiment_2")