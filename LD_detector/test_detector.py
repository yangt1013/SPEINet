import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from model.Speinet import SPEINet
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.nn as nn
import pickle
from tqdm import tqdm
import torchvision
from ptwt import wavedec2
import pywt
from torch.nn.functional import lp_pool2d, avg_pool2d, conv2d
from torch import Tensor

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


def load_single_sequence(frames):
    out = [imageio.imread(fname) for fname in frames]
    return out


import torch
import numpy as np
from typing import Optional

def generate_sequence(frames: list):
    output_frames = []

    output_frames = torch.stack([torch.from_numpy(frame).float() for frame in frames]).permute(0, 3, 1, 2)

    return output_frames


def sobel(img: Tensor):
    r'''
    Compute the sobol filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    device = img.device
    Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).to(device)
    Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).to(device)
    G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
    G = G.unsqueeze(1)
    x = conv2d(img, G, padding='same')
    out = torch.sum(x**2, dim=1, keepdim=True).sqrt()
    return out

def laplacian(img: Tensor):
    r'''
    Compute the laplacian filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    # laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    device = img.device
    laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    out = conv2d(img, laplacian_kernel, stride=1, padding=1)
    return out

def mask(img):
    r'''
    Compute the laplacian filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    device = img.device
    mask_size = torch.tensor([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    out = conv2d(img, mask_size)
    return out

def mis3_kernel(img: Tensor):
    r'''
    Compute the sobol filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    device = img.device
    Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]).to(device)
    Gy = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).to(device)
    grad_x = conv2d(img, Gx.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = conv2d(img, Gy.unsqueeze(0).unsqueeze(0), padding=1)
    out = torch.abs(grad_x) + torch.abs(grad_y)
    return out

def focus_measure_rmse_contrast(frames, kernel_size):
    '''
    frames is np array H x W containing a sequence of gray scale images
    '''
    c_bar = avg_pool2d(frames, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    avg_c = avg_pool2d((frames - c_bar)**2, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    contrast = lp_pool2d(torch.sqrt(avg_c), kernel_size=kernel_size, norm_type=2)**2
    return contrast.mean(dim=(1,2,3))

def focus_measure_gra0(frames, kernel_size:int):
    device = frames.device
    neighbor_size = 3
    neighbor_kernel = torch.ones((1, 1, neighbor_size, neighbor_size), device = device)
    contrast = conv2d(mis3_kernel(frames), neighbor_kernel, padding = neighbor_size//2)
    return lp_pool2d(contrast, norm_type=1, kernel_size=kernel_size).mean(dim=(1,2,3))

def focus_measure_mis3(frames, kernel_size:int):
    device = frames.device
    mis3_filter = torch.zeros((9,1,3,3), device=device)
    mis3_filter[:,:,1,1] = 1
    mis3_filter[0,0,0,0] = -1
    mis3_filter[1,0,0,1] = -1
    mis3_filter[2,0,0,2] = -1
    mis3_filter[3,0,1,0] = -1
    mis3_filter[4,0,1,1] = 0
    mis3_filter[5,0,1,2] = -1
    mis3_filter[6,0,2,0] = -1
    mis3_filter[7,0,2,1] = -1
    mis3_filter[8,0,2,2] = -1
    contrast = conv2d(frames ,mis3_filter, padding=1).abs().sum(dim=1, keepdim=True)
    return lp_pool2d(contrast, norm_type=1, kernel_size=kernel_size).mean(dim=(1,2,3))

def focus_measure_gra7(frames, kernel_size):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    g = sobel(frames)
    g_bar = avg_pool2d(g, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    phi = lp_pool2d((g - g_bar), kernel_size=kernel_size, norm_type=2)**2
    return phi.mean(dim=(1,2,3))


def focus_measure_lap1(frames, kernel_size: int):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    la = laplacian(frames)
    phi = lp_pool2d(la, norm_type=2, kernel_size=kernel_size)**2
    return phi.mean(dim=(1,2,3))

def focus_measure_wave1(frames: Tensor, kernel_size: int):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    wavelet = pywt.Wavelet('db6')
    W_LH1, W_HL1, W_HH1 = wavedec2(frames, wavelet, mode = 'zero', level=1)[1]
    phi = torch.sum((torch.abs(W_LH1)+ torch.abs(W_HL1) + torch.abs(W_HH1)), dim=(1,2,3))
    return phi

def focus_measure_sta3(frames, kernel_size):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    avg = avg_pool2d(frames, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    phi = lp_pool2d((frames - avg), kernel_size=kernel_size, norm_type=2)**2
    return phi.mean(dim=(1,2,3))

def focus_measure_dct3(frames, kernel_size:int):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    g = mask(frames)
    phi = lp_pool2d(g, kernel_size=kernel_size, norm_type=1)**2
    return phi.mean(dim=(1,2,3))

def generate_vars(frames: Tensor, kernel_size: int, device):
    r'''
    An incredibly long descritpion of this script with the link to the paper
    '''
    frames = frames.to(device)
    gray_frames = torchvision.transforms.Grayscale()(frames)/255.
    lap = focus_measure_lap1(gray_frames, kernel_size)
    mis3 = focus_measure_mis3(gray_frames, kernel_size)
    wave1 = focus_measure_wave1(gray_frames, kernel_size)
    gra_7 = focus_measure_gra7(gray_frames, kernel_size)
    sta_3 = focus_measure_sta3(gray_frames, kernel_size)
    dct_3 = focus_measure_dct3(gray_frames, kernel_size)
    return lap, mis3, wave1, gra_7, sta_3, dct_3


class Inference:
    def __init__(self, args):

        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        
        self.result_path = args.result_path
        self.n_seq = args.n_sequence
        self.size_must_mode = 4
        self.device = 'cuda'
        self.GPUs = args.n_GPUs

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        self.input_path = os.path.join(self.data_path, "blur")
        self.GT_path = os.path.join(self.data_path, "gt")
        self.label_path = os.path.join(self.data_path, "labelt")
        print('label_path', self.label_path)

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = SPEINet(in_channels=3, n_sequence=self.n_seq, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda', args=args)
        self.net.load_state_dict(torch.load(self.model_path))  #, strict=False
        self.net = self.net.to(self.device)
        if args.n_GPUs > 1:
            self.net = nn.DataParallel(self.net, range(args.n_GPUs))
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()

    def collate_all_vars(self, framess, kernel_size: int, device):
        all_vars = []
        frames = load_single_sequence(framess)
        blurry_sequence = generate_sequence(frames)
        vars_tuple = generate_vars(blurry_sequence, kernel_size, device)
        all_vars.append(torch.stack(vars_tuple, dim=1))
        variables = torch.cat(all_vars, dim=0).cpu().numpy()
        return variables



    def infer(self):
        with torch.no_grad():

            videos = sorted(os.listdir(self.input_path))
            
            for v in videos:
                start_time = time.time()
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*")))
                label_GT = np.load(os.path.join(self.label_path, v + ".npy"))

                # Check if precomputed labels exist
                label_file_path = os.path.join(self.label_path, v + "_labels.npy")
                if os.path.exists(label_file_path):
                    labels = np.load(label_file_path)
                else:
                    variables = self.collate_all_vars(input_frames, kernel_size=11, device=self.device)
                    with open('/home/yangt/nas/video_deblurring_nas/sy32/code/detector/pickle/LogisticRegression_0.5_11.pkl', 'rb') as f:
                        model1 = pickle.load(f)
                    labels = model1.predict(variables)

                # Ensure labels is not empty
                if len(labels) == 0:
                    self.logger.write_log(f"Warning: No labels generated for video {v}. Skipping accuracy calculation.")
                    continue

                result = sum(1 for m, n in zip(labels, label_GT) if (m == n))
                acc = result / len(labels)
                name = os.path.split(os.path.dirname(input_frames[0]))[-1]
                print(name + " Avg_acc: %.6f" % acc)
                end_time = time.time() 
                video_time = end_time - start_time  
                print(f"Time taken to process video {v}: {video_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDVD-TSP-Inference')

    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--border', action='store_true', default=True, help='restore border images of video if true')
    parser.add_argument('--default_data', type=str, default='BSDtest_all',
                        help='quick test, optional: REDS, GOPRO')
    parser.add_argument('--data_path', type=str, default='../dataset/DVD/test',
                        help='the path of test data')
    parser.add_argument('--model_path', type=str, default='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
                        help='the path of pretrain model')
    parser.add_argument('--window_size', type=int, default=5, help='window size used in the Swin Transformer')  # Add this line
    parser.add_argument('--result_path', type=str, default='../infer_results',
                        help='the path of deblur result')
    args = parser.parse_args()
    if args.default_data == 'GOPRO':
        args.data_path ='/home/yangt/ssd1/speinet_dataset/test/0.5_testblur/test'#'/home/yangt/ssd1/STDAT/dataset/Gopro_Random/test'
        args.model_path = '/home/yangt/nas/video_deblurring_nas/sy32/experiment/gpafinall-lunwenoutput-beifen/model/model_best.pt'
        args.result_path = '/home/yangt/ssd1/video_deblurring_ssd1/huatu/detector_result'
        args.n_colors = 3
        args.n_sequence = 3
        args.patch_size = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.rgb_range = 1
        args.n_GPUs = 2


    Infer = Inference(args)
    Infer.infer()
