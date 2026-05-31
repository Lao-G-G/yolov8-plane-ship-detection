import os
import xml.etree.ElementTree as ET
import shutil
import math
from PIL import Image

# ==================== 配置 ====================
NWPU_ROOT = 'path/to/NWPU'
HRSC_ROOT = 'path/to/HRSC2016'
OUTPUT_ROOT = './combined_dataset'

# NWPU: 0=airplane, 1=ship
# 统一映射: 0=ship, 1=airplane
NWPU_CLASS_MAP = {
    0: 1,  # airplane -> 1
    1: 0,  # ship -> 0
}

# 创建输出目录
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_ROOT, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', split), exist_ok=True)


# ==================== NWPU 转换 ====================
def convert_nwpu():
    """NWPU 已经是 YOLO 格式，直接复制并过滤类别"""

    for split in ['train', 'val', 'test']:
        img_src = os.path.join(NWPU_ROOT, split, 'images')
        lbl_src = os.path.join(NWPU_ROOT, split, 'labels')

        if not os.path.exists(img_src):
            print(f"NWPU {split} 不存在，跳过")
            continue

        img_files = [f for f in os.listdir(img_src) if f.endswith(('.jpg', '.png', '.jpeg'))]

        valid_count = 0
        for img_file in img_files:
            base = os.path.splitext(img_file)[0]
            lbl_path = os.path.join(lbl_src, base + '.txt')

            if not os.path.exists(lbl_path):
                continue

            # 读取标注，过滤类别
            filtered_lines = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])

                    # 只保留 ship(1) 和 airplane(0)
                    if cls_id in NWPU_CLASS_MAP:
                        new_cls = NWPU_CLASS_MAP[cls_id]
                        filtered_lines.append(f"{new_cls} {' '.join(parts[1:])}")

            # 只复制有目标的图像
            if filtered_lines:
                shutil.copy2(
                    os.path.join(img_src, img_file),
                    os.path.join(OUTPUT_ROOT, 'images', split, f'nwpu_{img_file}')
                )
                with open(os.path.join(OUTPUT_ROOT, 'labels', split, f'nwpu_{base}.txt'), 'w') as f:
                    f.write('\n'.join(filtered_lines))
                valid_count += 1

        print(f"NWPU {split}: {valid_count} 张有效")


# ==================== HRSC2016 转换 ====================
def parse_hrsc_xml(xml_path):
    """解析 HRSC2016 XML，返回所有旋转框"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        return None, None, []

    width = int(size.find('width').text)
    height = int(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue

        try:
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)

            boxes.append({
                'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle
            })
        except:
            continue

    return width, height, boxes


def convert_hrsc():
    """HRSC2016 转 YOLO 水平框"""

    img_dir = os.path.join(HRSC_ROOT, 'AllImages')
    anno_dir = os.path.join(HRSC_ROOT, 'Annotations')

    for split in ['train', 'val', 'test']:
        list_file = os.path.join(HRSC_ROOT, 'ImageSets', f'{split}.txt')
        if not os.path.exists(list_file):
            print(f"HRSC2016 {split}.txt 不存在")
            continue

        with open(list_file, 'r') as f:
            names = [line.strip().replace('.bmp', '') for line in f.readlines() if line.strip()]

        valid_count = 0
        for name in names:
            xml_path = os.path.join(anno_dir, f'{name}.xml')

            if not os.path.exists(xml_path):
                continue

            result = parse_hrsc_xml(xml_path)
            if result[0] is None:
                continue

            img_w, img_h, boxes = result

            if not boxes:
                continue

            # 找图像文件
            img_path = None
            for ext in ['.bmp', '.jpg', '.png']:
                candidate = os.path.join(img_dir, f'{name}{ext}')
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if not img_path:
                continue

            try:
                img = Image.open(img_path)
            except:
                continue

            # 转换所有旋转框为水平框
            lines = []
            for box in boxes:
                cx, cy, w, h, angle = box['cx'], box['cy'], box['w'], box['h'], box['angle']

                cos_a, sin_a = math.cos(angle), math.sin(angle)
                hw, hh = w / 2, h / 2

                corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                xs, ys = [], []
                for x, y in corners:
                    xr = x * cos_a - y * sin_a + cx
                    yr = x * sin_a + y * cos_a + cy
                    xs.append(xr)
                    ys.append(yr)

                xmin = max(0, min(xs))
                ymin = max(0, min(ys))
                xmax = min(img_w, max(xs))
                ymax = min(img_h, max(ys))

                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                bw = (xmax - xmin) / img_w
                bh = (ymax - ymin) / img_h

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                bw = max(0, min(1, bw))
                bh = max(0, min(1, bh))

                lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            if lines:
                output_img = os.path.join(OUTPUT_ROOT, 'images', split, f'hrsc_{name}.jpg')
                img.convert('RGB').save(output_img, 'JPEG')

                output_label = os.path.join(OUTPUT_ROOT, 'labels', split, f'hrsc_{name}.txt')
                with open(output_label, 'w') as f:
                    f.write('\n'.join(lines))

                valid_count += 1

        print(f"HRSC2016 {split}: {valid_count}/{len(names)} 张有效")


# ==================== 创建 data.yaml ====================
def create_yaml():
    yaml_content = f"""path: {OUTPUT_ROOT.replace('\\', '/')}
train: images/train
val: images/val
test: images/test

names:
  0: ship
  1: airplane
"""
    yaml_path = os.path.join(OUTPUT_ROOT, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\ndata.yaml 已创建: {yaml_path}")


# ==================== 统计 ====================
def print_stats():
    print("\n=== 最终数据集统计 ===")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(OUTPUT_ROOT, 'images', split)
        if os.path.exists(img_dir):
            files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            nwpu_count = len([f for f in files if f.startswith('nwpu_')])
            hrsc_count = len([f for f in files if f.startswith('hrsc_')])
            print(f"{split}: {len(files)} 张 (NWPU: {nwpu_count}, HRSC: {hrsc_count})")
        else:
            print(f"{split}: 目录不存在")


# ==================== 执行 ====================
if __name__ == '__main__':
    print("=== 转换 NWPU ===")
    convert_nwpu()

    print("\n=== 转换 HRSC2016 ===")
    convert_hrsc()

    print("\n=== 创建配置文件 ===")
    create_yaml()

    print("\n=== 统计 ===")
    print_stats()

    print(f"\n输出目录: {OUTPUT_ROOT}")
