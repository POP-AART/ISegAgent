import torch

def get_available_gpu_count():
    try:
        return torch.cuda.device_count()
    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    gpu_count = get_available_gpu_count()
    if gpu_count > 0:
        print(f"Found {gpu_count} GPU(s) available.")
    else:
        print("No GPU found.")

import torchio as tio

# 加载一个示例图像，这里假设您的图像路径为 image_path
image_path = '/var/scratch/jliu5/Medical/Datasets/data/train/tumor/pancreas/imagesTr/pancreas_train_002.nii.gz'
image = tio.ScalarImage(image_path)

# 获取像素间距
spacing = image.spacing
print("Pixel Spacing:", spacing)
"""
(0.7792969942092896, 0.7792969942092896, 7.5)
(0.7910159826278687, 0.7910159826278687, 5.0)

(3.0, 0.68359375, 0.68359375)
(1.0, 0.83203125, 0.83203125)

(0.8500000238418579, 0.8500000238418579, 4.0)
(0.84765625, 0.84765625, 1.0)

(0.7988280057907104, 0.7988280057907104, 2.5)
(0.703125, 0.703125, 5.0)

"""

import os
import torchio as tio


def get_nii_spacing(image_path):
    # 加载图像
    image = tio.ScalarImage(image_path)

    # 获取像素间距
    spacing = image.spacing

    return spacing


def traverse_folder(folder_path):
    # 遍历文件夹下的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 仅处理nii.gz文件
            if file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                spacing = get_nii_spacing(file_path)
                if (spacing) == (0.9765620231628418, 0.9765620231628418, 2.5):
                    print(f"File: {file}, Spacing: {(spacing)}")


# 指定文件夹路径
folder_path = '/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits/imagesTr'

# 调用遍历函数
# traverse_folder(folder_path)








