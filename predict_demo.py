from ultralytics import YOLO

model = YOLO('runs/detect/project_name/weights/best.pt')

results = model.predict(
    source='test_image.jpg',
    conf=0.25,
    iou=0.45,
    save=True,
    show=True,
    line_width=2,
    font_size=12
)

# print result
class_names = [
    'Airplane', 'Ship', 'Storage Tank', 'Baseball Diamond',
    'Tennis Court', 'Basketball Court', 'Ground Track Field',
    'Harbor', 'Bridge', 'Vehicle'
]

for result in results:
    boxes = result.boxes
    print(f"detected {len(boxes)} targets")
    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        print(f"  → {class_names[cls_id]} (confidence: {conf:.3f})")
