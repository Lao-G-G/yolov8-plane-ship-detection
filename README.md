# yolov8-plane-ship-detection

基于 YOLOv8m 的遥感图像目标检测项目，检测舰船、飞机和机场跑道。

## 项目结构
yolov8-plane-ship-detection/               
├── data/                  
│   ├── images/                 
│   │   ├── train/                 
│   │   ├── val/                 
│   │   └── test/                 
│   └── labels/                 
│       ├── train/                 
│       ├── val/                 
│       └── test/                 
├── data.yaml               
├── models/                 
├── runs/                   
├── train.py               
├── combine_data.py               
├── predict.py             
└── README.md             

## plugin1
仅使用NWPU_VHR10训练模型，由于此数据集仅包含ship和plane，就只看这两个参数，其他类别的指数忽略。详细数据如下:
### 训练结果

#### 损失曲线
![损失曲线](images/results.png)

#### 各类别 PR 曲线
![PR曲线](images/BoxPR_curve.png)

#### 混淆矩阵
![混淆矩阵](images/confusion_matrix.png)

### 预测示例
![预测示例](images/val_batch0_pred.jpg)
