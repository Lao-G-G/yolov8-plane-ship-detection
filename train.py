from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model_name = "yolov8m.pt"
    project_name = "nwpu_vhr2_hrsc"

    # load model
    model = YOLO(model_name)

    # train model
    results = model.train(
        data='combined_data.yaml',
        epochs=150,
        imgsz=640,
        batch=8,
        name=project_name,
        device=device,
        patience=30,                # Early Stopping
        save=True,
        plots=True,                 # Plot Results
        exist_ok=False,
        workers=8,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,

        # augmentation
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,               # ±15° Rotation
        translate=0.1,
        scale=0.5,
        flipud=0.0,                 # Flip Upside Down
        fliplr=0.5,                 # Flip Left and Right
        mosaic=1.0,                 # Mosaic
        mixup=0.1
    )

    metrics = model.val(data='combined_data.yaml', split='test')
    print("\n✅ NWPU test results:")
    print(f"  mAP@0.5 (all):      {metrics.box.map50:.5f}")
    print(f"  mAP@0.5:0.95:       {metrics.box.map:.5f}")
    print(f"  Precision:          {metrics.box.mp:.5f}")
    print(f"  Recall:             {metrics.box.mr:.5f}")

if __name__ == '__main__':
    main()
