import datetime
import logging
import copy
import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


sys.path.append(os.path.join(""))

from recommendation.agent_datasets.agent_dataset import load_agent_dataset
from recommendation.utils.utils_agent import agent_business, select_next_frame, recommend_frame, select_next_non_empty_frame
from recommendation.utils.utils_seg import finetune_model_predict3D
from recommendation.utils.misc import load_agent_checkpoint, save_agent_checkpoint
from recommendation.utils.utils_seg import compute_iou, compute_dice, compute_nsd, generate_metrics_of_each_slice
from recommendation.models.agent import Agent

# SAM
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry


# IVOS-W
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
from utils.data_paths import img_datas, img_datas2

from easydict import EasyDict as edict
from collections import OrderedDict, defaultdict

# SAM-Med3D
from utils.data_loader import Dataset_Union_ALL_Val

from train import build_model, get_dataloaders
"""
TODO: 1.produce reward
TODO: 2.pretrain agent
TODO: 3.train agent
"""

wandb.init(project="medsam3D_agent_train1", name="training pre_trained union agent")

parser = argparse.ArgumentParser()


parser.add_argument('--task_name', type=str, default='pretrain_agent')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=True)
parser.add_argument('--checkpoint', type=str, default='/var/scratch/jliu5/Medical/Datasets/initmodel/sam_vit_b2.pth') # /var/scratch/jliu5/Medical/Datasets/
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--work_dir', type=str, default='/var/scratch/jliu5/Medical/Datasets/initmodel')

parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('-dt', '--data_type', type=str, default='Ts')
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--multi_gpu', action='store_true', default=False)  # TODO: do not support true, has init problem
parser.add_argument('--num_workers', type=int, default=4)
# parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('-cp', '--checkpoint_path', type=str, default='/var/scratch/jliu5/Medical/Datasets/initmodel/union_train/sam_model_latest_with_pretrain.pth')
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-vp', '--vis_path', type=str, default='None')
parser.add_argument('-nc', '--num_clicks', type=int, default=5)  # should be 1 to generate the data for agent, then it increases
parser.add_argument('-pm', '--point_method', type=str, default='random_from_slices')

# data setting
parser.add_argument('-tdp', '--train_data_path', type=str, default='/var/scratch/jliu5/Medical/Datasets/data/train/tumor/lits')
parser.add_argument('--agent_checkpoint_name', type=str, default='agent_lits_epoch15_pretrain.pt')
parser.add_argument('--num_agent_epochs', type=int, default=15)

# interaction setting
parser.add_argument('--phase', type=str, default='train')  # TODO: pretrain, train, eval
parser.add_argument('--max_nb_interactions', type=int, default=5)

parser.add_argument('--seed', type=int, default=2024)


# init
args = parser.parse_args()
device = args.device
click_methods = {
    # 'default': get_next_click3D_torch_ritm,
    # 'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
    'random_from_slices': random_from_slices
}

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# TODO: change the config and args, but configs are not changed, so consider it later
agent_config = edict({
    'agent': {
        'memory_size': 100000,
        'gamma': 0.95,
        'eps_start': 0.7,
        'eps_end': 0.25,
        'eps_decay': 500,
        'update_rate': 0.05,
        'lr': 0.000005,
        'weight_decay': 0.000000000000005, 
        'sample_th': 0.01,  # TODO: 要改
        'train_batch_size': 32,
        'reward_csv': 'reward.csv',
        'pretrain_csv': 'pretrain.csv',
        'save_result_dir': '/var/scratch/jliu5/Medical/SAM-Med3D/recommendation/agent_data/pretrained/iou'
    },
    'data': {
        'subset': 'train',
        'num_workers': 2,
        'root_dir_davis': 'not important'
    },
    'method': 'ours', # ours | worst | random | linspace
    'setting': 'oracle', # oracle | wild
    'dataset': 'davis',
    'phase': 'train', # baseline, pretrain, train, eval
    'ckpt_dir': '/var/scratch/jliu5/Medical/SAM-Med3D/recommendation/weights/pretrained/iou',
})

print(agent_config.agent)

# TODO: main loop
def train_agent():

    # agent_config.method = 'random'
    agent = Agent(device=device, cfg=agent_config)
    weight_name = args.agent_checkpoint_name
    load_agent_checkpoint(agent, agent_config.ckpt_dir, device=device, strict=True, weight_name=weight_name)
    # agent.memory_pool.basename_csv = agent_config.agent.reward_csv
    agent.to(device)
    assess_net = None
    
    save_result_dir = agent_config.agent.save_result_dir

    # TODO: main changings
    path_to_reward = os.path.join(agent_config.agent.save_result_dir, agent_config.agent.reward_csv)
    path_to_pretrain = os.path.join(agent_config.agent.save_result_dir, agent_config.agent.pretrain_csv)
    df = pd.read_csv(path_to_reward, index_col=0)
    
    agent.memory_pool.load_from_csv(path_to_pretrain, agent_config.agent.save_result_dir, agent_config.agent.sample_th)
    seq_list = agent.memory_pool.seq_list
    print(f"init memory pool from {path_to_pretrain}, now the size of memory pool is: {len(agent.memory_pool)}")
    

    # all_dataset_paths = [args.train_data_path]
    if args.train_data_path is None:
        all_dataset_paths = img_datas2
    else:
        all_dataset_paths = [args.train_data_path]

    print(all_dataset_paths)


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

    # TODO: metrics
    all_iou_list = []
    all_dice_list = []
    all_nsd_list = []

    out_dice = dict()
    out_dice_all = OrderedDict()

    out_nsd = dict()
    out_nsd_all = OrderedDict()

    csv_name = 'metrics_train.csv' 
    if not os.path.exists(os.path.join(save_result_dir, csv_name)):
        metrics_df = pd.DataFrame(columns=['sequence', 'step', 'n_interaction', 'iou', 'dice', 'nsd'])
        metrics_df.to_csv(os.path.join(save_result_dir, csv_name), index=False)

    metrics_df = pd.read_csv(os.path.join(save_result_dir, csv_name))

    for batch_data in tqdm(test_dataloader):
        step = 0

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
        # TODO: 目前是随机初始化默认有一切片是有了标注的，实际上不是
        annotated_frames = []

        

        max_nb_interactions = args.max_nb_interactions
        num_agent_epochs = args.num_agent_epochs
        for n_interaction_i in range(1, num_agent_epochs*(max_nb_interactions) + 1):  # TODO: 12 can be changed
            
            n_interaction = n_interaction_i % max_nb_interactions  # to prevent the csv file then get nan when calculating the std
            if n_interaction == 0:
                n_interaction = max_nb_interactions

            if n_interaction % max_nb_interactions == 1:
                
                first_scribble = True
                old_masks_metric = None
                new_masks_metric, _ , _ = generate_metrics_of_each_slice(image3D, gt3D)
                old_frame = None
                old_masks_meta = None
                seen_seq[sequence] = 1 if sequence not in seen_seq.keys() else seen_seq[sequence] + 1
                repeat_selection = None
                # df = None  # TODO: should be deleted
                next_frame = select_next_non_empty_frame(gt3D)
                annotated_frames.append(next_frame)
                annotated_frames_list = [next_frame]
                first_frame = annotated_frames[0]

                reward_step_acc = 0
                reward_done_acc = 0

                old_masks_meta = None
                new_masks_meta = None

                # rec_kwargs
                prev_frames = [next_frame]
                n_frame = sz[-3]
                n_object = 1
                all_F = None
                mask_quality = None

                if (seen_seq[sequence] - 1) % 3 == 0:  # TODO: don't sure why 3, but followed this settings.
                    agent_dataset = load_agent_dataset(agent_config, agent.memory_pool.seq_list)
                agent_train_loader = DataLoader(
                    agent_dataset,
                    batch_size=agent_config.agent.train_batch_size,
                    shuffle=True,
                    num_workers=agent_config.data.num_workers,
                    pin_memory=True)
            else:
                annotated_frames_list_np = np.zeros(len(new_masks_metric))
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


            

            # ------frame recommendation------
            next_frame = recommend_frame(cfg_yl=agent_config, assess_net=assess_net, agent=agent, device=device,
                                         n_frame=n_frame, n_objects=n_object, all_F=all_F, all_P=all_P, 
                                         new_masks_quality=new_masks_metric, prev_frames=prev_frames,
                                         annotated_frames_list=copy.deepcopy(annotated_frames_list), mask_quality=mask_quality,
                                         first_frame=first_frame, max_nb_interactions=max_nb_interactions,
                                         gt3D=gt3D
                                         )

            prev_frames.append(next_frame)
            # ------agent------
            
            new_masks_meta = dict(
                sequence=sequence, scribble_iter=seen_seq[sequence], 
                n_interaction=n_interaction)


            [agent_loss_iter, reward_step, reward_done] = \
                    agent_business(phase=args.phase, agent=agent, max_nb_interactions=max_nb_interactions, 
                                   n_interaction=n_interaction, first_scribble=first_scribble, 
                                   old_masks_metric=old_masks_metric, new_masks_metric=new_masks_metric, 
                                   old_frame=old_frame, sequence=sequence, seen_seq=seen_seq, repeat_selection=repeat_selection, 
                                   df=df, annotated_frames_list=annotated_frames_list, next_frame=next_frame, 
                                   old_masks_meta=old_masks_meta, new_masks_meta=new_masks_meta, report_save_dir=save_result_dir,
                                   agent_train_loader=agent_train_loader)
            
            reward_step_acc += reward_step
            reward_done_acc += reward_done

            # ------metrics------
            print(f"sequence: {sequence}, seen_seq: {seen_seq[sequence]}, n_interaction: {n_interaction}, next_frame: {next_frame}, reward_step: {reward_step}, reward_done: {reward_done}")
            print(f"next_frame: {next_frame:2d} [{int(sum(new_masks_metric < new_masks_metric[next_frame])) + 1:2d}/{new_masks_metric.shape[0]:2d}]\t"
                    f"reward_step:{reward_step:.2f}  \t"
                    f"reward_done:{reward_done:.2f}  \t"
                    f"seq: {sequence}_{seen_seq[sequence]:1d} [{n_interaction:2d}/{max_nb_interactions:2d}]\t")
            # print iou, dice, nsd
            print(f"iou: {max(iou_list):.2f}, dice: {max(dice_list):.2f}, nsd: {max(nsd_list):.2f}")
            print("-------------------------------------")
            # break

            
            step += 1  # the number of interactions, there are many interactions in one sequence/instance/object
            

            all_iou_list.append(max(iou_list))
            all_dice_list.append(max(dice_list))
            all_nsd_list.append(max(nsd_list))
            print(dice_list)

            # dice
            out_dice[img_name] = max(dice_list)
            cur_dice_dict = OrderedDict()
            for i, dice in enumerate(dice_list):
                cur_dice_dict[f'{i}'] = dice
            out_dice_all[img_name[0]] = cur_dice_dict

            # nsd
            out_nsd[img_name] = max(nsd_list)
            cur_nsd_dict = OrderedDict()
            for i, nsd in enumerate(nsd_list):
                cur_nsd_dict[f'{i}'] = nsd
            out_nsd_all[img_name[0]] = cur_nsd_dict


            metrics_row = {
                'sequence': sequence.split('_', 1)[0],
                'step': step,
                'n_interaction': n_interaction,
                'iou': max(iou_list),
                'dice': max(dice_list),
                'nsd': max(nsd_list),
            }
            metrics_df = metrics_df.append(metrics_row, ignore_index=True)

            metrics_df.to_csv(os.path.join(save_result_dir, csv_name))

            wandb.log({"iou_per_step": max(iou_list),"dice_per_step": max(dice_list),"nsd_per_step": max(nsd_list)})

        wandb.log({"iou": max(iou_list),"dice": max(dice_list),"nsd": max(nsd_list)})

        save_agent_checkpoint(agent.policy_net, ckpt_dir=agent_config.ckpt_dir, weight_name=weight_name)
    
    save_agent_checkpoint(agent.policy_net, ckpt_dir=agent_config.ckpt_dir.replace('weights', 'weights_final'), weight_name=weight_name)

if __name__ == '__main__':
    train_agent()
    print("\033[91m------ends-----\033[0m")