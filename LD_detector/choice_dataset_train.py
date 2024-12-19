import torch
from torch.utils.data import Dataset
from random import choice
import os
import imageio.v3 as imageio
import random
import numpy as np
from PIL import Image
import shutil
import json

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_path_train = '/home/yangt/ssd1/dataset/GOPRO_Large_all/train'
output_dir = '/home/yangt/ssd1/dataset/GoProRS'
output_file = '/home/yangt/ssd1/dataset/GoProRS.json'

def find_subdirs_with_name_sharp(dir_path):
    return sorted(os.listdir(dir_path))

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

class GoProrc(Dataset):
    def __init__(self, dir_path, ratio) -> None:
        super().__init__()
        self.dir_path = dir_path
        self.subdirs = find_subdirs_with_name_sharp(dir_path)
        self.ratio = ratio
        self.description = f"GoPro_{float(ratio)}"
    
    def __len__(self):
        return len(self.subdirs)

    def _load_single_sequence_(self, dir_path):
        list_images = sorted(os.listdir(dir_path))
        list_fname_path = [os.path.join(dir_path, frame) for frame in list_images]
        frames = [imageio.imread(name) for name in list_fname_path if name.lower().endswith(('png', 'jpg', 'jpeg'))]
        return frames

    def _generate_blurry_sequence_(self, frames, window_range=(1, 15), threshold=5, seed=None):
        blurry_frames = []
        gt_frames = []
        frames_labels = []

        while len(frames) != 0:
            random_label = torch.tensor(int((random.random() < self.ratio) or (len(frames) <= threshold))).to(device)
            frames_labels.append(random_label)
            if random_label:
                window_size = random.randint(window_range[0], threshold)
            else:
                window_size = random.randint(threshold + 1, window_range[1])

            window = frames[:window_size]
            frames = frames[window_size:]
            blurry_frame = torch.from_numpy(np.mean(window, axis=0)).float().to(device)
            gt_frame = torch.from_numpy(window[len(window) // 2]).float().to(device)

            gt_frames.append(gt_frame)
            blurry_frames.append(blurry_frame)
        
        blurry_frames = torch.stack(blurry_frames).permute(0, 3, 1, 2)
        gt_frames = torch.stack(gt_frames).permute(0, 3, 1, 2)
        frames_labels = torch.stack(frames_labels)

        return blurry_frames, frames_labels, gt_frames
    
    def __getitem__(self, idx):
        video = self._load_single_sequence_(os.path.join(self.dir_path, self.subdirs[idx]))
        blurry_sequence, labels, gt_sequence = self._generate_blurry_sequence_(video)
        return {'blurry_sequence': blurry_sequence, 'labels': labels, 'gt_sequence': gt_sequence}

def process_dataset(dataset, data_idx, mode):
    loc_dataset = choice(dataset)
    print(f"Using dataset: {loc_dataset.description} for {mode}")
    trainset = loc_dataset[data_idx]
    vid_input_names = trainset['blurry_sequence']
    vid_gt_names = trainset['gt_sequence']
    vid_label_names = trainset['labels']
    print(f'{mode} set', len(trainset['blurry_sequence']))
    
    subdir_name = loc_dataset.subdirs[data_idx]
    blur_subdir_path = os.path.join(output_dir, mode, 'blur', subdir_name)
    gt_subdir_path = os.path.join(output_dir, mode, 'gt', subdir_name)
    labels_dir_path = os.path.join(output_dir, mode, 'label')
    
    os.makedirs(blur_subdir_path, exist_ok=True)
    os.makedirs(gt_subdir_path, exist_ok=True)
    os.makedirs(labels_dir_path, exist_ok=True)

    blurry_paths = []
    gt_paths = []

    for i, (blurry_frame, gt_frame) in enumerate(zip(vid_input_names, vid_gt_names)):
        blurry_image_path = os.path.join(blur_subdir_path, f'{i:04d}.png')
        gt_image_path = os.path.join(gt_subdir_path, f'{i:04d}.png')
        
        blurry_image = blurry_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt_image = gt_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        Image.fromarray(blurry_image).save(blurry_image_path)
        Image.fromarray(gt_image).save(gt_image_path)

        blurry_paths.append(blurry_image_path)
        gt_paths.append(gt_image_path)

    labels_path = os.path.join(labels_dir_path, f'{subdir_name}.npy')
    np.save(labels_path, vid_label_names.cpu().numpy())
    print(f'Saved labels to {labels_path}')
    
    return blurry_paths, gt_paths, vid_label_names.cpu().numpy()

def save_output_to_file(blurry_paths_train, gt_paths_train, labels_train, output_file):
    output_data = {
        "Train Blurry Paths": blurry_paths_train,
        "Train GT Paths": gt_paths_train,
        "Train Labels": [label.tolist() for label in labels_train]  # Convert numpy arrays to lists
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def generate_datasets():
    clear_directory(output_dir)

    gopro_0_train = GoProrc(dir_path_train, 0.05)
    gopro_25_train = GoProrc(dir_path_train, 0.25)
    gopro_5_train = GoProrc(dir_path_train, 0.5)
    gopro_mix_train = [gopro_0_train, gopro_25_train, gopro_5_train]

    all_blurry_paths_train = []
    all_gt_paths_train = []
    all_labels_train = []

    for data_idx in range(len(gopro_0_train)):
        torch.cuda.empty_cache()  # Free up any cached memory
        blurry_paths, gt_paths, labels = process_dataset(gopro_mix_train, data_idx, 'test')
        all_blurry_paths_train.append(blurry_paths)
        all_gt_paths_train.append(gt_paths)
        all_labels_train.append(labels)

    save_output_to_file(all_blurry_paths_train, all_gt_paths_train, all_labels_train, output_file)
    print(f"Output saved to {output_file}")

generate_datasets()
