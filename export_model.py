from ultralytics import YOLO

model = YOLO('runs/detect/nwpu_vhr10_detection/weights/best.pt')

# ONNX
model.export(format='onnx', imgsz=640, simplify=True)

# TensorRT
model.export(format='engine', imgsz=640, half=True, device=0)

# OpenVINO
model.export(format='openvino', imgsz=640)
