img_datas = [
'/var/scratch/jliu5/Medical/Datasets/new_data/train/tumor/pancreas',
'/var/scratch/jliu5/Medical/Datasets/data/train/tumor/kits',
'/var/scratch/jliu5/Medical/Datasets/data/train/tumor/colon',
'/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits'
]

img_datas2 = [
'/var/scratch/jliu5/Medical/Datasets/data/train/tumor/pancreas',
'/var/scratch/jliu5/Medical/Datasets/data/train/tumor/kits',
'/var/scratch/jliu5/Medical/Datasets/data/train/tumor/colon',
'/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits'
]

img_val_datas = [
'/var/scratch/jliu5/Medical/Datasets/new_data/validation/tumor/pancreas',
'/var/scratch/jliu5/Medical/Datasets/data/validation/tumor/kits',
'/var/scratch/jliu5/Medical/Datasets/data/validation/tumor/colon',
'/var/scratch/jliu5/Medical/Datasets/data/validation/tumor/lits'
]

all_classes = [
'tumor'
]

'''

(medsam) [jliu5@node406 SAM-Med3D]$ python relabel_dataset2.py
['lits_train_047.nii.gz', 'lits_train_023.nii.gz', 'lits_train_009.nii.gz', 'lits_train_081.nii.gz', 'lits_train_054.nii.gz', 'lits_train_053.nii.gz', 'lits_train_020.nii.gz', 'lits_train_094.nii.gz', 'lits_train_008.nii.gz', 'lits_train_066.nii.gz', 'lits_train_026.nii.gz', 'lits_train_012.nii.gz', 'lits_train_035.nii.gz', 'lits_train_056.nii.gz', 'lits_train_088.nii.gz', 'lits_train_064.nii.gz', 'lits_train_007.nii.gz', 'lits_train_087.nii.gz', 'lits_train_027.nii.gz', 'lits_train_039.nii.gz', 'lits_train_034.nii.gz', 'lits_train_092.nii.gz', 'lits_train_059.nii.gz', 'lits_train_013.nii.gz', 'lits_train_075.nii.gz', 'lits_train_093.nii.gz', 'lits_train_091.nii.gz', 'lits_train_070.nii.gz', 'lits_train_055.nii.gz', 'lits_train_019.nii.gz', 'lits_train_005.nii.gz', 'lits_train_043.nii.gz', 'lits_train_049.nii.gz', 'lits_train_002.nii.gz', 'lits_train_069.nii.gz', 'lits_train_084.nii.gz', 'lits_train_032.nii.gz', 'lits_train_031.nii.gz', 'lits_train_038.nii.gz', 'lits_train_010.nii.gz', 'lits_train_067.nii.gz', 'lits_train_037.nii.gz', 'lits_train_065.nii.gz', 'lits_train_063.nii.gz', 'lits_train_078.nii.gz', 'lits_train_076.nii.gz', 'lits_train_071.nii.gz', 'lits_train_079.nii.gz', 'lits_train_028.nii.gz', 'lits_train_011.nii.gz', 'lits_train_021.nii.gz', 'lits_train_044.nii.gz', 'lits_train_016.nii.gz', 'lits_train_062.nii.gz', 'lits_train_015.nii.gz', 'lits_train_042.nii.gz', 'lits_train_058.nii.gz', 'lits_train_080.nii.gz', 'lits_train_018.nii.gz', 'lits_train_060.nii.gz', 'lits_train_036.nii.gz', 'lits_train_082.nii.gz', 'lits_train_074.nii.gz', 'lits_train_033.nii.gz', 'lits_train_040.nii.gz', 'lits_train_004.nii.gz', 'lits_train_090.nii.gz', 'lits_train_006.nii.gz', 'lits_train_051.nii.gz', 'lits_train_024.nii.gz', 'lits_train_072.nii.gz', 'lits_train_003.nii.gz', 'lits_train_048.nii.gz', 'lits_train_046.nii.gz', 'lits_train_052.nii.gz', 'lits_train_029.nii.gz', 'lits_train_001.nii.gz', 'lits_train_085.nii.gz', 'lits_train_073.nii.gz', 'lits_train_041.nii.gz', 'lits_train_022.nii.gz', 'lits_train_050.nii.gz', 'lits_train_077.nii.gz', 'lits_train_083.nii.gz', 'lits_train_057.nii.gz', 'lits_train_061.nii.gz', 'lits_train_030.nii.gz', 'lits_train_014.nii.gz', 'lits_train_045.nii.gz', 'lits_train_017.nii.gz', 'lits_train_068.nii.gz', 'lits_train_086.nii.gz', 'lits_train_025.nii.gz']
File: lits_train_075.nii.gz
Image Spacing: (0.85546875, 0.85546875, 0.699999988079071)
Label Spacing: (0.8554688096046448, 0.8554688096046448, 0.699999988079071)
Spacing is inconsistent!

File: lits_train_065.nii.gz
Image Spacing: (0.68359375, 0.68359375, 0.699999988079071)
Label Spacing: (0.6835938096046448, 0.6835938096046448, 0.699999988079071)
Spacing is inconsistent!

File: lits_train_021.nii.gz
Image Spacing: (0.74609375, 0.74609375, 0.699999988079071)
Label Spacing: (0.7460938096046448, 0.7460938096046448, 0.699999988079071)
Spacing is inconsistent!

File: lits_train_051.nii.gz
Image Spacing: (0.90234375, 0.90234375, 3.0)
Label Spacing: (1.0, 1.0, 1.0)
Spacing is inconsistent!

File: lits_train_086.nii.gz
Image Spacing: (0.69921875, 0.69921875, 0.6999999284744263)
Label Spacing: (0.6992188096046448, 0.6992188096046448, 0.6999999284744263)
Spacing is inconsistent!

File: lits_train_075.nii.gz
Image Spacing: (0.85546875, 0.85546875, 0.699999988079071)
Label Spacing: (0.8554688096046448, 0.8554688096046448, 0.699999988079071)
Spacing is inconsistent. Fixing...

File: lits_train_065.nii.gz
Image Spacing: (0.68359375, 0.68359375, 0.699999988079071)
Label Spacing: (0.6835938096046448, 0.6835938096046448, 0.699999988079071)
Spacing is inconsistent. Fixing...

File: lits_train_021.nii.gz
Image Spacing: (0.74609375, 0.74609375, 0.699999988079071)
Label Spacing: (0.7460938096046448, 0.7460938096046448, 0.699999988079071)
Spacing is inconsistent. Fixing...

File: lits_train_051.nii.gz
Image Spacing: (0.90234375, 0.90234375, 3.0)
Label Spacing: (1.0, 1.0, 1.0)
Spacing is inconsistent. Fixing...

File: lits_train_086.nii.gz
Image Spacing: (0.69921875, 0.69921875, 0.6999999284744263)
Label Spacing: (0.6992188096046448, 0.6992188096046448, 0.6999999284744263)
Spacing is inconsistent. Fixing...

'''