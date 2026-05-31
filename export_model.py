from ultralytics import YOLO

model = YOLO('runs/detect/nwpu_vhr10_detection/weights/best.pt')

# 导出为 ONNX（通用）
model.export(format='onnx', imgsz=640, simplify=True)

# 导出为 TensorRT（NVIDIA GPU / Jetson）
model.export(format='engine', imgsz=640, half=True, device=0)

# 导出为 OpenVINO（Intel CPU）
model.export(format='openvino', imgsz=640)
