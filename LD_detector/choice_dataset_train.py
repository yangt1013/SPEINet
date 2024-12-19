import torch
from torch import nn
from torch.utils.data import Dataset
from random import choice
import os
import imageio.v3 as imageio
import random
from typing import Optional, Tuple
import numpy as np
from PIL import Image

dir_path = '/home/yangt/ssd1/dataset/train'
output_dir = '/home/yangt/ssd1/dataset/newblur'

def find_subdirs_with_name_sharp(dir_path):
    return sorted(os.listdir(dir_path))

class GoProrc(Dataset):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.subdirs = find_subdirs_with_name_sharp(dir_path)
        self.ratio = ratio
        self.description = f"GoPro_{float(ratio)}"
    
    def __len__(self):
        return len(self.subdirs)

    def _load_single_sequence_(self, dir_path):
        list_images = sorted(os.listdir(dir_path))
        list_fname_path = [os.path.join(dir_path, frame) for frame in list_images]
        frames = [imageio.imread(name) for name in list_fname_path]
        return frames

    def _generate_blurry_sequence_(self, frames, window_range=(1, 15), threshold=5, seed=None):
        blurry_frames = []
        gt_frames = []
        frames_labels = []

        while len(frames) != 0:
            random_label = torch.tensor(int((random.random() < self.ratio) or (len(frames) <= threshold)))
            frames_labels.append(random_label)
            if random_label:
                window_size = random.randint(window_range[0], threshold)
            else:
                window_size = random.randint(threshold + 1, window_range[1])

            window = frames[:window_size]
            frames = frames[window_size:]
            blurry_frame = torch.from_numpy(np.mean(window, axis=0)).float()
            gt_frame = torch.from_numpy(window[len(window) // 2]).float()

            gt_frames.append(gt_frame)
            blurry_frames.append(blurry_frame)
        
        blurry_frames = torch.stack(blurry_frames).permute(0, 3, 1, 2)
        gt_frames = torch.stack(gt_frames).permute(0, 3, 1, 2)
        frames_labels = torch.stack(frames_labels)

        return blurry_frames, frames_labels, gt_frames
    
    def __getitem__(self, idx):
        video = self._load_single_sequence_(os.path.join(dir_path, self.subdirs[idx]))
        blurry_sequence, labels, gt_sequence = self._generate_blurry_sequence_(video)
        return {'blurry_sequence': blurry_sequence, 'labels': labels, 'gt_sequence': gt_sequence}

def process_dataset(dataset, data_idx):
    loc_dataset = choice(dataset)
    print(f"Using dataset: {loc_dataset.description}")
    trainset = loc_dataset[data_idx]
    vid_input_names = trainset['blurry_sequence']
    vid_gt_names = trainset['gt_sequence']
    vid_label_names = trainset['labels']
    print('trainset', len(trainset['blurry_sequence']))

    subdir_name = loc_dataset.subdirs[data_idx]
    subdir_path = os.path.join(output_dir, subdir_name)
    blurry_subdir_path = os.path.join(subdir_path, 'blurry')
    gt_subdir_path = os.path.join(subdir_path, 'gt')
    
    os.makedirs(blurry_subdir_path, exist_ok=True)
    os.makedirs(gt_subdir_path, exist_ok=True)

    blurry_paths = []
    gt_paths = []

    for i, (blurry_frame, gt_frame) in enumerate(zip(vid_input_names, vid_gt_names)):
        blurry_image_path = os.path.join(blurry_subdir_path, f'blurry_frame_{i}.png')
        gt_image_path = os.path.join(gt_subdir_path, f'gt_frame_{i}.png')
        
        # convert to numpy arrays
        blurry_image = blurry_frame.permute(1, 2, 0).numpy().astype(np.uint8)
        gt_image = gt_frame.permute(1, 2, 0).numpy().astype(np.uint8)

   
        Image.fromarray(blurry_image).save(blurry_image_path)
        Image.fromarray(gt_image).save(gt_image_path)

    
        blurry_paths.append(blurry_image_path)
        gt_paths.append(gt_image_path)

    labels_path = os.path.join(subdir_path, 'labels.npy')
    np.save(labels_path, vid_label_names.numpy())
    print(f'Saved labels to {labels_path}')
    
    return blurry_paths, gt_paths, vid_label_names.numpy()

def generate_datasets():
    gopro_0 = GoProrc(0.05)
    gopro_25 = GoProrc(0.25)
    gopro_5 = GoProrc(0.5)
    gopro_mix = [gopro_0, gopro_25, gopro_5]

    all_blurry_paths = []
    all_gt_paths = []
    all_labels = []

    for data_idx in range(len(gopro_0)):
        blurry_paths, gt_paths, labels = process_dataset(gopro_mix, data_idx)
        all_blurry_paths.append(blurry_paths)
        all_gt_paths.append(gt_paths)
        all_labels.append(labels)

    return all_blurry_paths, all_gt_paths, all_labels, blurry_paths, gt_paths, labels


all_blurry_paths, all_gt_paths, all_labels, blurry_paths, gt_paths, labels = generate_datasets()

print("Blurry Paths:", all_blurry_paths)
print("GT Paths:", all_gt_paths)
print("Labels:", all_labels)
