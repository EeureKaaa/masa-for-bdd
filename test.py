import math
import os.path
import argparse

import torch
import torchvision.utils as vutils
import gc

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from data import *

from pathlib import Path
current_script_path = Path(__file__).resolve()
base_path = current_script_path.parent.parent.parent.parent


import sys
sys.path.append(str(base_path))

# from abstraction_model.generate_slot_with_mask_input.dynamicrafter.abstractionNet_slot_with_mask_flatten_dynamicrafter_fourier_pos_emb_action_slot_FSQ import STEVE as AbstractionNet
# from util.utils import cosine_anneal, linear_warmup
# import wandb
import os
from generate_parallel_lt import masking

# os.environ['CUDA_VISIBLE_DEVICES']="7"
def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)
parser = argparse.ArgumentParser()

#for mask generation
parser.add_argument('--video_dir', type=str, default="/home/wxh/data_wxh/hhz/bdd_seg/videos/val", help='Directory containing input videos.')
parser.add_argument('--output_dir', type=str, default="/home/wxh/data_wxh/hhz/bdd_seg_masa/bonding_box/clip/val", help='Directory to save output data.')
parser.add_argument('--config_file', type=str, default="/home/wxh/data_wxh/hhz/masa/configs/masa-gdino/masa_gdino_swinb_inference.py", help='Path to the YOLOX config file.')
parser.add_argument('--checkpoint_file', type=str, default="/home/wxh/data_wxh/hhz/masa/saved_models/masa_models/gdino_masa.pth", help='Path to the YOLOX checkpoint file.')
# parser.add_argument('--config_file', type=str, default="/home/wxh/data_wxh/hhz/masa/projects/mmdet_configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py", help='Path to the YOLOX config file.')
# parser.add_argument('--checkpoint_file', type=str, default="/data/wxh/from_xftp/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth", help='Path to the YOLOX checkpoint file.')
parser.add_argument('--score_thr', type=float, default=0.15, help='Score threshold for detections.')
parser.add_argument('--target_width', type=int, default=128, help='Target width for frame resizing.')
parser.add_argument('--image_width', type=int, default=128, help='Target width for frame resizing.')
parser.add_argument('--num_sub_process_per_gpu', type=int, default=2, help='Number of subprocesses per GPU.')
parser.add_argument('--max_object_num', type=int, default=64)
parser.add_argument('--ep_len', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--video_ext', type=str, default=".mp4", help='video file type.')
parser.add_argument('--phase', type=str, default="test", help='video file type.')


parser.add_argument('--use_masa', type=str2bool, default=False)

args = parser.parse_args()

import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # mask_buffer_train = initialize_mask_buffer(100)
    print(args, args.video_dir, args.phase)
    masa_dataset = MasaDataset(args, args.video_dir, phase=args.phase)
    # args.output_dir = os.path.join(args.output_dir, args.phase)
    # val_dataset = VideoDataset(args.video_dir, mask_buffer_val, transform=None)
    masa_loader_kwargs = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': True,
    }

    masa_loader = DataLoader(masa_dataset, sampler=None, **masa_loader_kwargs)
    # val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)

    # mask_buffer_val = initialize_mask_buffer(100)
    if args.use_masa:
        sample_list_train = list(masa_loader.sampler)
        masking(args, None, sample_list_train, masa_dataset, args.phase)
    sample_list_val = list(val_loader.sampler)

    masking(args, None, sample_list_train, masa_dataset, args.phase)



    ###### masking(args, mask_buffer_val, sample_list_val, val_dataset, 'val')
    for epoch in range(10):
        train_dataset = LanguageTable_Dataset(args, args.output_dir, 'test', args.image_width, 4,img_glob='*.png',
                 image_mode=False, random_clip =False, return_all_mask=False)
        loader_kwargs = {
            'batch_size': 1#args.batch_size,
            # 'shuffle': True,
            # 'num_workers': 4,
            # 'pin_memory': True,
            # 'drop_last': True,
        }
        train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
        for batch in train_loader:
            # print(i)
            # print(batch)
            import time

            # Pause the execution for 5 seconds
            time.sleep(5)
            # break
