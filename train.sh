python train.py --seed 2023\
 --vp ./Datasets/data/train/tumor/colon \
 --cp ./Datasets/initmodel/sam_med3d_turbo.pth \
 --save_name ./Datasets/results/sam_vit_b.py \
 --tdp ./data/validation_test1 -nc 10 \
 --image_size 128 \
 --mt vit_b \
 --dim 2 \
 --save_name ./results/colon/union_out_dice.py