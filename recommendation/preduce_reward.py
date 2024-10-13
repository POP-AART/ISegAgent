import datetime
import logging
import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(""))

from recommendation.utils.utils_agent import agent_business, select_next_frame, select_next_non_empty_frame
from recommendation.utils.utils_seg import finetune_model_predict3D
from recommendation.utils.utils_seg import compute_iou, compute_dice, compute_nsd, generate_metrics_of_each_slice
from recommendation.models.agent import Agent

from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry
from tqdm import tqdm
from torch.backends import cudnn
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2, random_from_slices
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas2

from easydict import EasyDict as edict
from collections import OrderedDict, defaultdict
from utils.data_loader import Dataset_Union_ALL_Val
from train import build_model, get_dataloaders



parser = argparse.ArgumentParser()

parser.add_argument('--task_name', type=str, default='pretrain_agent')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=True)
parser.add_argument('--checkpoint', type=str, default='/var/scratch/jliu5/Medical/Datasets/initmodel/sam_vit_b2.pth') # the checkpoint of the model, sam_vit_b2.pth is the model trained from scratch
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--work_dir', type=str, default='/var/scratch/jliu5/Medical/Datasets/initmodel')
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('-dt', '--data_type', type=str, default='Ts')
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--multi_gpu', action='store_true', default=False)  # TODO: do not support true, has init problem
parser.add_argument('--num_workers', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('-cp', '--checkpoint_path', type=str, default='/var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest_with_pretrain.pth')
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-vp', '--vis_path', type=str, default='None')
parser.add_argument('-nc', '--num_clicks', type=int, default=5)  # should be 1 to generate the data for agent, then it increases
parser.add_argument('-pm', '--point_method', type=str, default='random_from_slices')

parser.add_argument('--test_data_path', type=str, default='/var/scratch/jliu5/Medical/Datasets/data/train/tumor/pancreas')
parser.add_argument('--phase', type=str, default='baseline')  # pretrain, train, eval
parser.add_argument('--max_nb_interactions', type=int, default=5)  # this should be consistent at all stages
parser.add_argument('--num_interaction_epochs', type=int, default=5)

parser.add_argument('--seed', type=int, default=2024)

args = parser.parse_args()
device = args.device
click_methods = {
    # 'default': get_next_click3D_torch_ritm,
    # 'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
    'random_from_slices': random_from_slices,
}

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

agent_config = edict({
    'agent': {
        'memory_size': 100000,
        'gamma': 0.95,
        'eps_start': 0.7,
        'eps_end': 0.25,
        'eps_decay': 500,
        'update_rate': 0.05,
        'lr': 0.000005,
        'weight_decay': 0.0005, 
        'reward_csv': 'pretrained/reward.csv',
    },
    'data': {
        'subset': 'train',
    }
})

print(agent_config.agent)


def generate_rewards():

    
    agent = Agent(device=device, cfg=agent_config)
    agent.memory_pool.basename_csv = agent_config.agent.reward_csv
    agent.to(device)


    all_dataset_paths = img_datas2

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size, args.crop_size, args.crop_size)),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Tr", 
        data_type=args.data_type, 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )
    

    checkpoint_path = args.checkpoint_path
    if (args.dim == 3):
        sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
        if checkpoint_path is not None:
            model_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = model_dict['model_state_dict']
            sam_model_tune.load_state_dict(state_dict)
            print(f"Load model from {checkpoint_path}")
    elif (args.dim == 2):
        pass  # add the 2D model here

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)

    # metrics
    all_iou_list = []
    all_dice_list = []
    all_nsd_list = []

    out_dice = dict()
    out_dice_all = OrderedDict()

    out_nsd = dict()
    out_nsd_all = OrderedDict()

    for batch_data in tqdm(test_dataloader):
        image3D, gt3D, spacing, img_name = batch_data
        sz = image3D.size()
        if(sz[2]<args.crop_size or sz[3]<args.crop_size or sz[4]<args.crop_size):  # False
            print("[ERROR] wrong size", sz, "for", img_name)
        
        modality = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_name[0]))))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(img_name[0])))
        vis_root = os.path.join(os.path.dirname(__file__), args.vis_path, modality, dataset)
        pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"))
        
        sequence = img_name[0].split('/')[-1].split('.')[0]
        seen_seq = {}
        annotated_frames = []

        
        max_nb_interactions = args.max_nb_interactions
        for n_interaction_i in range(1, 9*(max_nb_interactions) + 1):  
            
            n_interaction = n_interaction_i % max_nb_interactions
            if n_interaction == 0:
                n_interaction = max_nb_interactions

            if n_interaction % max_nb_interactions == 1:
                agent_train_loader = None

                first_scribble = True
                old_masks_metric = None
                
                old_frame = None
                old_masks_meta = None
                seen_seq[sequence] = 1 if sequence not in seen_seq.keys() else seen_seq[sequence] + 1
                repeat_selection = None
                df = None
                next_frame = select_next_non_empty_frame(gt3D)

                annotated_frames.append(next_frame)
                annotated_frames_list = [next_frame]
                first_frame = annotated_frames[0]

                reward_step_acc = 0
                reward_done_acc = 0

                old_masks_meta = None
                new_masks_meta = None
            else:
                annotated_frames_list_np = np.zeros(len(new_masks_metric))
                # next_frame = random.randint(0, image3D.shape[-1] - 1)
                for i in annotated_frames_list:
                    annotated_frames_list_np[i] += 1
                repeat_selection = next_frame not in list(
                    np.where(annotated_frames_list_np == annotated_frames_list_np.min())[0])
                annotated_frames_list.append(next_frame)

                first_scribble = False
                old_frame = next_frame
                old_masks_metric = new_masks_metric
                old_masks_meta = new_masks_meta

            # -----segmentation-----
            norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            # 3D prediction
            # TODO: 加一个参数，每次交互次数和交互多少个帧，每个帧加一个随机click
            if(args.dim==3):
                seg_mask_list, points, labels, iou_list, dice_list, nsd_list, all_P = finetune_model_predict3D(
                    image3D, gt3D, spacing, sam_model_tune, device=device,
                    click_method=args.point_method, num_clicks=len(annotated_frames_list), 
                    prev_masks=None, crop_size=args.crop_size, 
                    click_methods=click_methods,
                    slices=annotated_frames_list,
                    norm_transform=norm_transform)
            elif(args.dim==2):
                pass  # TODO: add the 2D model here

            seg_mask_final = torch.from_numpy(seg_mask_list[-1]).unsqueeze(0).unsqueeze(0)
            new_masks_metric, _ , _ = generate_metrics_of_each_slice(seg_mask_final, gt3D)

        
            # ------frame recommendation------
            next_frame = random.randint(0, image3D.shape[-1] - 1)
        
            next_frame = select_next_non_empty_frame(gt3D)
      
            # ------agent------
            save_result_dir = os.path.join(os.getcwd(), "recommendation", 'agent_data')
            new_masks_meta = dict(
                sequence=sequence, scribble_iter=seen_seq[sequence], 
                n_interaction=n_interaction)


            [agent_loss_iter, reward_step, reward_done] = \
                    agent_business(phase=args.phase, agent=agent, max_nb_interactions=max_nb_interactions, 
                                   n_interaction=n_interaction, first_scribble=first_scribble, 
                                   old_masks_metric=old_masks_metric, new_masks_metric=new_masks_metric, 
                                   old_frame=old_frame, sequence=sequence, seen_seq=seen_seq, repeat_selection=repeat_selection, df=df, annotated_frames_list=annotated_frames_list, next_frame=next_frame, old_masks_meta=old_masks_meta, new_masks_meta=new_masks_meta, report_save_dir=save_result_dir,
                                   agent_train_loader=None)
            
            reward_step_acc += reward_step
            reward_done_acc += reward_done

            # ------metrics------
            print(f"sequence: {sequence}, seen_seq: {seen_seq[sequence]}, n_interaction: {n_interaction}, next_frame: {next_frame}, reward_step: {reward_step}, reward_done: {reward_done}")
            print(f"annotated_frames_list: {annotated_frames_list}")
            print("-------------------------------------")
            # break


if __name__ == '__main__':
    generate_rewards()
    print("\033[91m------ends-----\033[0m")