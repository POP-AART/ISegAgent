import pickle
import os
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


origin_image_path = '/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits/imagesTr'
origin_label_path = '/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits/labelsTr'

new_image_path = '/var/scratch/jliu5/Medical/Datasets/new_data/train/tumor/lits/imagesTr'
new_label_path = '/var/scratch/jliu5/Medical/Datasets/new_data/train/tumor/lits/labelsTr'

if not os.path.exists(new_image_path):
    os.makedirs(new_image_path)

if not os.path.exists(new_label_path):
    os.makedirs(new_label_path)

# start
file_list = os.listdir(origin_image_path)

if 'colon' in origin_image_path:
    target_class = 1
else:
    target_class = 2

for file in tqdm(file_list):
    sitk_image = sitk.ReadImage(os.path.join(origin_image_path, file))
    image = sitk.GetArrayFromImage(sitk_image)

    sitk_label = sitk.ReadImage(os.path.join(origin_label_path, file))
    label = sitk.GetArrayFromImage(sitk_label)

    image[np.isnan(image)] = 0
    label[np.isnan(label)] = 0

    label = (label == target_class).astype(np.float32)

    new_sitk_image = sitk.GetImageFromArray(image)
    new_sitk_image.SetOrigin(sitk_image.GetOrigin())
    new_sitk_image.SetSpacing(sitk_image.GetSpacing())
    new_sitk_image.SetDirection(sitk_image.GetDirection())

    new_sitk_label = sitk.GetImageFromArray(label)
    new_sitk_label.SetOrigin(sitk_label.GetOrigin())
    new_sitk_label.SetSpacing(sitk_label.GetSpacing())
    new_sitk_label.SetDirection(sitk_label.GetDirection())

    sitk.WriteImage(new_sitk_image, os.path.join(new_image_path, file))
    sitk.WriteImage(new_sitk_label, os.path.join(new_label_path, file))

print('done')
