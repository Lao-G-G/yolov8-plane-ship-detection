from ultralytics import YOLO

model = YOLO('runs/detect/nwpu_vhr10_detection/weights/best.pt')

results = model.predict(
    source='test_image.jpg',
    conf=0.25,          # 遥感小目标建议降低阈值
    iou=0.45,
    save=True,
    show=True,
    line_width=2,
    font_size=12
)

# 打印检测结果
class_names = [
    'Airplane', 'Ship', 'Storage Tank', 'Baseball Diamond',
    'Tennis Court', 'Basketball Court', 'Ground Track Field',
    'Harbor', 'Bridge', 'Vehicle'
]

for result in results:
    boxes = result.boxes
    print(f"检测到 {len(boxes)} 个目标")
    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        print(f"  → {class_names[cls_id]} (置信度: {conf:.3f})")
