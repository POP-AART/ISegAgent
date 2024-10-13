# import pickle
# import os
# import shutil
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# import numpy as np
#
# image_path = '/var/scratch/jliu5/Medical/Datasets/data/train/tumor/kits/imagesTr/kits_train_001.nii.gz'
# label_path = '/var/scratch/jliu5/Medical/Datasets/data/train/tumor/kits/labelsTr/kits_train_001.nii.gz'
#
# sitk_image = sitk.ReadImage(image_path)
# sitk_label = sitk.ReadImage(label_path)
#
# image = sitk.GetArrayFromImage(sitk_image)
# label = sitk.GetArrayFromImage(sitk_label)
#
# for i in range(label.shape[2]):
#     if label[:,:,i].sum()>0:
#         print(i)
#
# print(np.unique(label))
#
# slice_index = 40  # 例如，选择第50个切片
#
# # 创建一个包含1行2列的子图
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#
# # 绘制图像切片
# axes[0].imshow(image[:,:,slice_index], cmap='gray')
# axes[0].set_title(f'Image Slice {slice_index}')
# axes[0].set_xlabel('X')
# axes[0].set_ylabel('Y')
#
# # 绘制掩码切片
# axes[1].imshow(label[:,:,slice_index], cmap='gray')  # 使用jet色彩图显示掩码
# axes[1].set_title(f'Mask Slice {slice_index}')
# axes[1].set_xlabel('X')
# axes[1].set_ylabel('Y')
#
# plt.show()

# import os
# import SimpleITK as sitk
#
# def get_spacing(file_path):
#     image = sitk.ReadImage(file_path)
#     return image.GetSpacing()
#
# def compare_spacing(images_folder, labels_folder):
#     image_files = os.listdir(images_folder)
#     label_files = os.listdir(labels_folder)
#
#     for filename in image_files:
#         if filename.endswith('.nii.gz'):
#             image_path = os.path.join(images_folder, filename)
#             label_path = os.path.join(labels_folder, filename)
#
#             if os.path.exists(label_path):
#                 image_spacing = get_spacing(image_path)
#                 label_spacing = get_spacing(label_path)
#
#
#
#                 if image_spacing == label_spacing:
#                     # print("Spacing is consistent.\n")
#                     pass
#                 else:
#                     print(f"File: {filename}")
#                     print(f"Image Spacing: {image_spacing}")
#                     print(f"Label Spacing: {label_spacing}")
#                     print("Spacing is inconsistent!\n")
#             else:
#                 print(f"Label file not found for {filename}\n")
#
# # 指定数据集路径
# data_folder = "/var/scratch/jliu5/Medical/Datasets/new_data/train/tumor/lits"
#
# # imagesTr 文件夹和 labelsTr 文件夹的完整路径
# images_folder = os.path.join(data_folder, "imagesTr")
# labels_folder = os.path.join(data_folder, "labelsTr")
#
# # 比较 spacing
# compare_spacing(images_folder, labels_folder)

import os

cpu_count = os.cpu_count()

print(f"Number of CPUs: {cpu_count}")
