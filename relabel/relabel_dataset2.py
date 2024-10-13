import os
import SimpleITK as sitk

image_filenames = []

def get_spacing(file_path):
    image = sitk.ReadImage(file_path)
    return image.GetSpacing()

def compare_images_spacing(images_folder, labels_folder):
    image_files = os.listdir(images_folder)
    label_files = os.listdir(labels_folder)

    print(image_files)

    for filename in image_files:
        if filename.endswith('.nii.gz'):
            image_path = os.path.join(images_folder, filename)
            label_path = os.path.join(labels_folder, filename)

            if os.path.exists(label_path):
                image_spacing = get_spacing(image_path)
                label_spacing = get_spacing(label_path)

                print(f"File: {filename}")
                print(f"Image Spacing: {image_spacing}")
                print(f"Label Spacing: {label_spacing}")

                if image_spacing == label_spacing:
                    # print("Spacing is consistent.\n")
                    pass
                else:
                    image_filenames.append(filename)

                    print("Spacing is inconsistent!\n")
            else:
                print(f"Label file not found for {filename}\n")


def compare_spacing(image_paths, label_paths):
    for image_path, label_path in zip(image_paths, label_paths):
        set_spacing(image_path, label_path)


def set_spacing(image_path, label_path):
    image_spacing = get_spacing(image_path)
    label_spacing = get_spacing(label_path)

    print(f"File: {os.path.basename(image_path)}")
    print(f"Image Spacing: {image_spacing}")
    print(f"Label Spacing: {label_spacing}")

    if image_spacing == label_spacing:
        print("Spacing is consistent.\n")
    else:
        print("Spacing is inconsistent. Fixing...\n")

        # 使用 Resample 函数修改 spacing
        label = sitk.ReadImage(label_path)
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(label.GetSize())
        resampler.SetOutputSpacing(image_spacing)
        resampled_label = resampler.Execute(label)

        # 保存修改后的标签
        sitk.WriteImage(resampled_label, label_path)


data_folder = "/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits"
images_folder = os.path.join(data_folder, "imagesTr")
labels_folder = os.path.join(data_folder, "labelsTr")
# image_filenames = ['lits_train_075.nii.gz', 'lits_train_065.nii.gz', 'lits_train_021.nii.gz', 'lits_train_086.nii.gz']


compare_images_spacing(images_folder, labels_folder)
# image_filenames.append('lits_train_089.nii.gz')
image_paths = [os.path.join(images_folder, filename) for filename in image_filenames]
label_paths = [os.path.join(labels_folder, filename) for filename in image_filenames]
compare_spacing(image_paths, label_paths)
print(image_paths)
print(label_paths)

# ------------------------- relabel_dataset2.py --------------------------------
# import os
# import SimpleITK as sitk
#
# image_filenames = ['']
#
# def get_spacing(file_path):
#     image = sitk.ReadImage(file_path)
#     return image.GetSpacing()
#
# def compare_images_spacing(images_folder, labels_folder):
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
#                 if image_spacing == label_spacing:
#                     # print("Spacing is consistent.\n")
#                     pass
#                 else:
#                     print(f"File: {filename}")
#                     image_filenames.append(filename)
#                     print(f"Image Spacing: {image_spacing}")
#                     print(f"Label Spacing: {label_spacing}")
#                     print("Spacing is inconsistent!\n")
#             else:
#                 print(f"Label file not found for {filename}\n")
#
# def set_spacing(image_path, label_path):
#     image_spacing = get_spacing(image_path)
#     label_spacing = get_spacing(label_path)
#
#     print(f"File: {os.path.basename(image_path)}")
#     print(f"Image Spacing: {image_spacing}")
#     print(f"Label Spacing: {label_spacing}")
#
#     if image_spacing == label_spacing:
#         print("Spacing is consistent.\n")
#     else:
#         print("Spacing is inconsistent. Fixing...\n")
#
#         # 使用 Resample 函数修改 spacing
#         label = sitk.ReadImage(label_path)
#         resampler = sitk.ResampleImageFilter()
#         resampler.SetSize(label.GetSize())
#         resampler.SetOutputSpacing(image_spacing)
#         resampled_label = resampler.Execute(label)
#
#         # 保存修改后的标签
#         sitk.WriteImage(resampled_label, label_path)
#
#
# # 指定数据集路径
# data_folder = "/var/scratch/jliu5/Medical/Datasets/new_data/train/tumor/lits"
# data_to_folder = "/var/scratch/jliu5/Medical/Datasets/new_data/train/tumor/tests"
#
# # imagesTr 文件夹和 labelsTr 文件夹的完整路径
# images_folder = os.path.join(data_folder, "imagesTr")
# labels_folder = os.path.join(data_folder, "labelsTr")
#
# # 比较 spacing
# compare_images_spacing(images_folder, labels_folder)
#
#
# '''
# modify spacing
# '''
#
# def set_spacing(image_path, label_path):
#     image_spacing = get_spacing(image_path)
#     label_spacing = get_spacing(label_path)
#
#     print(f"File: {os.path.basename(image_path)}")
#     print(f"Image Spacing: {image_spacing}")
#     print(f"Label Spacing: {label_spacing}")
#
#     if image_spacing == label_spacing:
#         print("Spacing is consistent.\n")
#     else:
#         print("Spacing is inconsistent. Fixing...\n")
#
#         # 使用 Resample 函数修改 spacing
#         image = sitk.ReadImage(image_path)
#         resampler = sitk.ResampleImageFilter()
#         resampler.SetSize(image.GetSize())
#         resampler.SetOutputSpacing(image_spacing)
#         resampled_image = resampler.Execute(image)
#
#         sitk.WriteImage(resampled_image, label_path)
#
# def compare_spacing(image_paths, label_paths):
#     for image_path, label_path in zip(image_paths, label_paths):
#         set_spacing(image_path, label_path)
#
# # 指定数据集路径
# data_folder = "/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits"
#
# # imagesTr 文件夹和 labelsTr 文件夹的完整路径
# images_folder = os.path.join(data_folder, "imagesTr")
# labels_folder = os.path.join(data_folder, "labelsTr")
#
# image_filenames = ['lits_train_075.nii.gz', 'lits_train_065.nii.gz', 'lits_train_021.nii.gz', 'lits_train_086.nii.gz']
#
#
# image_paths = [os.path.join(images_folder, filename) for filename in image_filenames]
# label_paths = [os.path.join(labels_folder, filename) for filename in image_filenames]
#
# # 比较 spacing 并修改
# compare_spacing(image_paths, label_paths)
