import os
import glob
import torch
import random
import numpy
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from collections import deque
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import numpy as np
from time import time, sleep
import torch.nn.functional as F
import os
import cv2
import torch
from torch.utils.data import Dataset
# from generate_parallel_DP import *

class MasaDataset(Dataset):
    def __init__(self, args, root_dir, mask_buffer=None, transform=None, phase='train', dataset_type='lt'):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'lt/').
            transform (callable, optional): Optional transform to be applied
                on a video sample.
        """
        self.phase=''
        self.args = args
        self.root_dir = os.path.join(root_dir, self.phase)
        self.mask_buffer = mask_buffer
        self.transform = transform
        self.video_paths = self._get_video_paths()
        self.img_size = args.image_width
        self.masks_dir = os.path.join(args.output_dir,self.phase)
        self.max_num_objects = args.max_object_num
        # if not transform:
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ]
        )
        self.process_transform = transforms.Compose(
            [
                transforms.Resize((self.args.target_width, self.args.target_width)),
                transforms.ToTensor()
            ]
        )
        self.dataset_type = dataset_type

    def _get_video_paths(self):
        """
        Traverse the directory structure to find all videos with the format bdd_seg/videos/val/*/rgb.mp4
        """
        video_paths = []
        for root in os.listdir(self.root_dir):
            files = os.listdir(os.path.join(self.root_dir, root))
            video_idx = root.split('/')[-1]
            image_path = os.path.join(self.args.output_dir, self.args.phase, f'video{video_idx}', 'image')
            mask_path = os.path.join(self.args.output_dir, self.args.phase, f'video{video_idx}', 'mask')
            
            
            if os.path.exists(image_path) and os.path.exists(mask_path):
                if len(os.listdir(image_path)) ==  len(os.listdir(mask_path)) and \
                    len(os.listdir(image_path)) > 0:
                    continue
            
            for file in files:
                if file == 'rgb.mp4':
                    video_paths.append(os.path.join(self.root_dir, root, file))
        return video_paths

    def __len__(self):
        """
        Returns the total number of videos in the dataset
        """
        return len(self.video_paths)

    def _load_video(self, video_path):
        """
        Load the video from the given path using OpenCV.
        Returns:
            frames (list of np.array): List of video frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def create_bbox_tensor(self, dict_list):
        
        img_size = dict_list[0]['image_size']
        for i in range(len(dict_list)):
            del dict_list[i]['image_size']

        # Create a set of unique object IDs
        unique_object_ids = set()
        for d in dict_list:
            unique_object_ids.update(d.keys())
        # unique_object_ids.remove('image_size')
        
        # Map each unique object ID to a unique index
        object_id_to_index = {obj_id: idx for idx, obj_id in enumerate(unique_object_ids)}
        N = len(unique_object_ids)
        T = len(dict_list)


        bbox_tensor = np.zeros((T, N, 4), dtype=np.int64)
        # Fill the mask tensor
        for t, d in enumerate(dict_list):
            for obj_id, bbox in d.items():
                n_idx = object_id_to_index[obj_id]
                # h, w = mask.shape
                bbox_tensor[t, n_idx] += bbox

        return img_size, object_id_to_index, len(unique_object_ids), bbox_tensor
    
    
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the video.
        
        Returns:
            sample (dict): A dictionary containing video frames and path.
                {
                    'video': frames (list of np.array),
                    'path': video_path (str)
                }
        """
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)
        mask_path = os.path.join(self.masks_dir, f'video{idx}','mask')
        
        while True:
            if os.path.exists(mask_path) and len(os.listdir(mask_path)) == len(frames):
                sorted_masks = []
                for i in range(len(frames)):
                    mask = np.load(os.path.join(mask_path, f'frame{i:06d}.npy'), allow_pickle=True)
                    sorted_masks.append(mask)
                break
            
            else:
                # while True:
                #     masks = extract_mask_from_buffer(self.mask_buffer, idx)
                #     if masks != [] and len(masks) == len(frames):
                #         break
                sleep(0.1)        
                # sorted_masks = sorted(masks, key=lambda x: x[1])
        nbbox_dict = [bbox.item() for bbox in sorted_masks]
        
        img_size, obj_map, num_objects, nbbox = self.create_bbox_tensor(nbbox_dict)
        
        if num_objects > self.max_num_objects:
            nbbox = nbbox[:, :self.max_num_objects]
            print("Warning: Number of objects exceeds the maximum number of objects.")
        elif num_objects < self.max_num_objects:
            padding_size = self.max_num_objects - num_objects
            padding = [(0, 0), (0, padding_size), (0, 0)]  # No padding on T and D dimensions, pad n -> N

            # Apply padding using np.pad
            nbbox = np.pad(nbbox, pad_width=padding, mode='constant', constant_values=0)
        else:
            pass
        
        
        
        scale_x = self.img_size / img_size[2]
        scale_y = self.img_size / img_size[1]
        
        nbbox = (nbbox * np.array([scale_x, scale_y, scale_x, scale_y]).reshape(1, 1, 4)).astype(np.uint8)
        video_masks = []
        for i in range(len(frames)):
            background = np.ones((self.img_size, self.img_size), dtype=np.uint8)
            masks = []
            for j in range(nbbox.shape[1]):
                x1,y1,x2,y2 = nbbox[i, j]
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                background[y1:y2, x1:x2] = 0
                mask[y1:y2, x1:x2] = 1
                # background[x1:x2, y1:y2] = 0
                # mask[x1:x2, y1:y2] = 1
                masks.append(self.transform(Image.fromarray(mask)).bool().float())
            masks.insert(0, self.transform(Image.fromarray(background)).bool().float())
            masks = torch.stack(masks)
            video_masks.append(masks)
        
        
        video_masks = torch.stack(video_masks)         
        video = torch.stack([self.transform(Image.fromarray(frame)) for frame in frames])
        # if self.transform:
        #     frames = self.transform(frames)

        sample = {'video': video, 'mask': video_masks}
        # apply_mask_to_video(video, video_masks, output_dir='masked_videos', fps=4)
        return sample

    def get_video(self, idx):
        """
        Args:
            idx (int): Index of the video.
        
        Returns:
            sample (dict): A dictionary containing video frames and path.
                {
                    'video': frames (list of np.array),
                    'path': video_path (str)
                }
        """
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)

        video = torch.stack([self.process_transform(Image.fromarray(frame)) for frame in frames])


        # sample = {'video': frames, 'path': video_path}
        return video
    
    def get_video_path(self, idx):
        """
        Args:
            idx (int): Index of the video.
        
        Returns:
            sample (dict): A dictionary containing video frames and path.
                {
                    'video': frames (list of np.array),
                    'path': video_path (str)
                }
        """
        video_path = self.video_paths[idx]
        # frames = self._load_video(video_path)

        # video = torch.stack([self.process_transform(Image.fromarray(frame)) for frame in frames])


        # # sample = {'video': frames, 'path': video_path}
        return video_path

class LanguageTable_Dataset(Dataset):
    def __init__(self, args, root, phase, img_size, ep_len=3, img_glob='*.png',
                 image_mode=False, random_clip =False, return_all_mask=False, **kwargs):
        self.root = root
        self.phase = phase
        self.img_size = img_size
        self.total_dirs = self.get_paths()
        self.ep_len = ep_len
        self.image_mode = image_mode
        self.random_clip = random_clip
        self.return_all_mask = return_all_mask

        # if phase == 'train':
        #     self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        # elif phase == 'val':
        #     self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        # elif phase == 'test':
        #     self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        # else:
        #     pass

        # chunk into episodes
        self.max_num_objects = args.max_object_num
        
        
        self.episodes = []
        for image_dir in self.total_dirs:
            frame_buffer = deque(maxlen=self.ep_len)
            image_paths = sorted(glob.glob(os.path.join(image_dir, img_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(list(frame_buffer))
                    # frame_buffer = []

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.episodes)
    
    def create_bbox_tensor(self, dict_list):
        
        img_size = dict_list[0]['image_size']
        for i in range(len(dict_list)):
            del dict_list[i]['image_size']

        # Create a set of unique object IDs
        unique_object_ids = set()
        for d in dict_list:
            unique_object_ids.update(d.keys())
        # unique_object_ids.remove('image_size')
        
        # Map each unique object ID to a unique index
        object_id_to_index = {obj_id: idx for idx, obj_id in enumerate(unique_object_ids)}
        N = len(unique_object_ids)
        T = len(dict_list)


        bbox_tensors = np.zeros((T, N, 4), dtype=np.float64)
        # Fill the mask tensor
        for t, d in enumerate(dict_list):
            for obj_id, bbox in d.items():
                n_idx = object_id_to_index[obj_id]
                # h, w = mask.shape
                bbox_tensors[t, n_idx] += bbox[:4]
        # sorted_array = np.array([bbox_tensor[bbox_tensor[:, 4].argsort()[::-1]] for bbox_tensor in bbox_tensors])

        # Step 2: Remove the 5th position from each sub-array
        # result_array = sorted_array[:, :, :4]

        return img_size, object_id_to_index, len(unique_object_ids), bbox_tensors
    
    def get_paths(self):
        # root = os.path.join(self.root, self.phase)
        root =self.root
        image_dirs = []
        for video_folder in os.listdir(os.path.join(root, self.phase)):
            if not os.path.isdir(os.path.join(root, self.phase, video_folder)):
                continue
            mask_dir = os.path.join(root, self.phase, video_folder, 'mask')
            image_dir = os.path.join(root, self.phase, video_folder, 'image')
            image_dirs.append(image_dir)
            if len(os.listdir(mask_dir)) != len(os.listdir(image_dir)):
                continue
            # image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        return image_dirs
        

    def __getitem__(self, idx):
        video = []
        nbbox_dict = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            # image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
            bbox_loc = img_loc.replace('image', 'mask').replace('png', 'npz')
            
            bbox_dict = np.load(os.path.join(bbox_loc), allow_pickle=True)['arr_0'].item()
            nbbox_dict.append(bbox_dict)
            
        img_size, obj_map, num_objects, nbbox = self.create_bbox_tensor(nbbox_dict)
        
        if num_objects > self.max_num_objects:
            nbbox = nbbox[:, :self.max_num_objects]
            print("Warning: Number of objects exceeds the maximum number of objects.")
        elif num_objects < self.max_num_objects:
            padding_size = self.max_num_objects - num_objects
            padding = [(0, 0), (0, padding_size), (0, 0)]  # No padding on T and D dimensions, pad n -> N

            # Apply padding using np.pad
            nbbox = np.pad(nbbox, pad_width=padding, mode='constant', constant_values=0)
        else:
            pass
        
        
        
        scale_x = self.img_size / img_size[2]
        scale_y = self.img_size / img_size[1]
        
        nbbox = (nbbox * np.array([scale_x, scale_y, scale_x, scale_y]).reshape(1, 1, 4)).astype(np.uint8)
        video_masks = []
        for i in range(len(self.episodes[idx])):
            background = np.ones((self.img_size, self.img_size), dtype=np.uint8)
            masks = []
            for j in range(nbbox.shape[1]):
                x1,y1,x2,y2 = nbbox[i, j]
                bbox_dict = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                background[y1:y2, x1:x2] = 0
                bbox_dict[y1:y2, x1:x2] = 1
                # background[x1:x2, y1:y2] = 0
                # mask[x1:x2, y1:y2] = 1
                masks.append(self.transform(Image.fromarray(bbox_dict)).bool().float())
            masks.insert(0, self.transform(Image.fromarray(background)).bool().float())
            masks = torch.stack(masks)
            video_masks.append(masks)
        
        video = torch.stack(video, dim=0)
        video_masks = torch.stack(video_masks)         
        background_bbox = torch.tensor([0, 0, self.img_size, self.img_size]).reshape(1,1,4).repeat(video_masks.shape[0], 1, 1)
        video_bbox = torch.cat((background_bbox,torch.tensor(nbbox).float()), dim=1)
        
        
        if self.random_clip:
            clip_choice = random.randint(2, self.ep_len - 1)
        else:
            clip_choice = self.ep_len - 1

        if self.image_mode:
            ref_video = video[:1]
            video = video[clip_choice:clip_choice + 1]
            if not self.return_all_mask:
                video_mask = video_mask[clip_choice:clip_choice + 1]
                video_bbox = video_bbox[clip_choice:clip_choice + 1]
                # video_status = video_status[clip_choice:clip_choice + 1]
            else:
                indexes = [0, clip_choice]
                video_mask = video_mask[indexes]
                video_bbox = video_bbox[indexes]
                # video_status = [video_status[i] for i in indexes]
            video = (ref_video, video)
        else:
            video = (video, video.clone())
            
            
            
        
        # apply_mask_to_video(video, video_masks, output_dir='masked_videos', fps=4)
        return video, (video_masks, video_bbox, 0, self.max_num_objects+1)



def apply_mask_to_video(video, video_masks, output_dir='masked_videos', fps=4):
    """
    Apply the masks for each object in video and save masked videos.

    Args:
        video (torch.Tensor): Input video tensor of shape (T, 3, H, W)
        video_masks (torch.Tensor): Masks for the video of shape (T, N, 1, H, W)
        output_dir (str): Directory to save the masked videos
        fps (int): Frames per second for the output video
    """
    T, C, H, W = video.shape  # Video dimensions: (T, 3, H, W)
    _, N, _, _, _ = video_masks.shape  # Mask dimensions: (T, N, 1, H, W)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert video tensor to numpy (T, H, W, C) and scale to [0, 255] for saving
    video_np = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # Iterate over each object (N objects in total)
    for obj_idx in range(N):
        # Prepare video writer for each object
        output_video_path = os.path.join(output_dir, f'object_{obj_idx+1}_masked_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 videos
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

        # Iterate over each frame (T frames in total)
        for t in range(T):
            frame = video_np[t]  # Get frame t: (H, W, C)
            mask = video_masks[t, obj_idx, 0].cpu().numpy()  # Get mask for object obj_idx at frame t: (H, W)

            # Apply mask to the frame (broadcasting mask to 3 channels)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))

            # Write the masked frame to the video
            video_writer.write(masked_frame)

        # Release the video writer for this object
        video_writer.release()

    print(f"Masked videos saved in '{output_dir}' directory.")