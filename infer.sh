#python validation.py --seed 2023\
# -vp ./results/vis_sam_med3d \
# -cp ./ckpt/sam_med3d.pth \
# -tdp ./data/validation_test1 -nc 10 \
# --save_name ./results/sam_med3d.py

#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/colon \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/sam_med3d_turbo.pth
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/kits \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/sam_med3d_turbo.pth
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/lits \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/sam_med3d_turbo.pth
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/pancreas \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/sam_med3d_turbo.pth

# train sam model with 1 click
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/colon \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 1
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/kits \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 1
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/lits \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 1
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/pancreas \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 1

#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/colon \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 1
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/kits \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 1
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/lits \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 1
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/pancreas \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 1

# train sam model with 5 click
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/colon \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 5
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/kits \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 5
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/lits \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 5
python validation.py \
  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/pancreas \
  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_loss_best.pth \
  --num_clicks 5

#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/colon \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 5
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/kits \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 5
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/lits \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 5
#python validation.py \
#  --test_data_path /var/scratch/jliu5/Medical/Datasets/data/validation/tumor/pancreas \
#  --checkpoint_path /var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest.pth \
#  --num_clicks 5
