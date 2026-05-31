from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")

    # 选择模型（遥感小目标多，推荐 yolov8m 或 yolov8l）
    model_name = "yolov8m.pt"   # 可选: yolov8s / yolov8l / yolov8x
    project_name = "nwpu_vhr2_hrsc"

    # 加载预训练模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data='combined_data.yaml',
        epochs=150,                 # 遥感数据复杂，建议充分训练
        imgsz=640,                  # NWPU 图像较小，640 足够；若需更高精度可用 1024
        batch=8,                   # 640 分辨率下 batch=8（32G 显存可设 32）
        name=project_name,
        device=device,
        patience=30,                # 早停机制（验证集 mAP 不升则停）
        save=True,
        plots=True,                 # 自动生成 PR 曲线、混淆矩阵、样本可视化
        exist_ok=False,
        workers=8,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,

        # 数据增强（关键！提升小目标和尺度变化鲁棒性）
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,               # ±15° 旋转（遥感视角多样）
        translate=0.1,
        scale=0.5,
        flipud=0.0,                 # 不上下翻转（地理目标方向敏感）
        fliplr=0.5,                 # 左右翻转安全
        mosaic=1.0,                 # Mosaic 增强（对小目标如 vehicle/airplane 极有效）
        mixup=0.1
    )

    # 在测试集上评估最终性能
    metrics = model.val(data='dataset.yaml', split='test')
    print("\n✅ NWPU VHR-10 测试集结果:")
    print(f"  mAP@0.5 (all):      {metrics.box.map50:.5f}")
    print(f"  mAP@0.5:0.95:       {metrics.box.map:.5f}")
    print(f"  Precision:          {metrics.box.mp:.5f}")
    print(f"  Recall:             {metrics.box.mr:.5f}")

if __name__ == '__main__':
    main()
