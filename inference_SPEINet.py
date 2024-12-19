import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from model.speinet import SPEINet
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.nn as nn
import pickle
from tqdm import tqdm
import torchvision
from ptwt import wavedec2
import pywt
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        # self.bm_path = os.path.join(self.data_path, "Blur_map")
        self.label_path = os.path.join(self.data_path, "label")
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

    def return_BlurryIndices(self, detect_result, dist=7):
        """find all Blurry frames and their pre/sub sharp frames."""
        SharpIndex_list = [i for i in range(len(detect_result)) if detect_result[i] == 1]
        # print("Length of blur frames: %d" % len([i for i in range(len(detect_result)) if detect_result[i] == 0]))
        pre_index = 0
        sub_index = 1
        PreSIndex_list, SubSIndex_list = [], []
        if len(SharpIndex_list) > 1:
            for i in range(len(detect_result)):
                if i < SharpIndex_list[pre_index]:
                    if SharpIndex_list[pre_index] - i < dist:
                        PreSIndex_list.append(SharpIndex_list[pre_index])
                        SubSIndex_list.append(SharpIndex_list[pre_index])
                    else:
                        PreSIndex_list.append(i - 2 if i > 1 else i)
                        SubSIndex_list.append(i + 2 if i < len(detect_result) - 2 else i)
                elif i == SharpIndex_list[pre_index]:
                    PreSIndex_list.append(i)
                    SubSIndex_list.append(i)  # SharpIndex_list[sub_index]
                elif i > SharpIndex_list[pre_index] and i < SharpIndex_list[sub_index]:
                    if i - SharpIndex_list[pre_index] < dist:
                        PreSIndex_list.append(SharpIndex_list[pre_index])
                    else:
                        PreSIndex_list.append(i - 2)
                    if SharpIndex_list[sub_index] - i < dist:
                        SubSIndex_list.append(SharpIndex_list[sub_index])
                    else:
                        SubSIndex_list.append(i + 2)
                elif i == SharpIndex_list[sub_index]:
                    pre_index = pre_index + 1
                    sub_index = sub_index + 1
                    if sub_index > len(SharpIndex_list) - 1:
                        sub_index = sub_index - 1
                        pre_index = pre_index - 1
                    PreSIndex_list.append(i)  # SharpIndex_list[pre_index]
                    SubSIndex_list.append(i)  # SharpIndex_list[sub_index]
                elif i > SharpIndex_list[sub_index]:
                    if i - SharpIndex_list[sub_index] < dist:
                        PreSIndex_list.append(SharpIndex_list[sub_index])
                        SubSIndex_list.append(SharpIndex_list[sub_index])
                    else:
                        PreSIndex_list.append(i - 2)
                        SubSIndex_list.append(i + 2 if i < len(detect_result) - 2 else i)
        else:
            for i in range(len(detect_result)):
                if i == 0:
                    PreSIndex_list.append(i)
                    SubSIndex_list.append(i + 1)
                elif i == len(detect_result) - 1:
                    PreSIndex_list.append(i - 1)
                    SubSIndex_list.append(i)  # SharpIndex_list[sub_index]
                else:
                    PreSIndex_list.append(i - 1)
                    SubSIndex_list.append(i + 1)
       
        pre_sindex_len = len(PreSIndex_list)
        sub_sindex_len = len(SubSIndex_list)
    
        for i in range(pre_sindex_len // 2):
            if PreSIndex_list[i] not in SharpIndex_list:
                PreSIndex_list[i] = pre_sindex_len -1
        
        for i in range(pre_sindex_len // 2, pre_sindex_len):
            if PreSIndex_list[i] not in SharpIndex_list:
                PreSIndex_list[i] = 0
        
        for i in range(sub_sindex_len // 2):
            if SubSIndex_list[i] not in SharpIndex_list:
                SubSIndex_list[i] = sub_sindex_len - 1
        
        for i in range(sub_sindex_len // 2, sub_sindex_len):
            if SubSIndex_list[i] not in SharpIndex_list:
                SubSIndex_list[i] = 0

        return PreSIndex_list, SubSIndex_list
    
    def collate_all_vars(self, framess, kernel_size: int, device):
        all_vars = []
        frames = load_single_sequence(framess)
        blurry_sequence = generate_sequence(frames)
        vars_tuple = generate_vars(blurry_sequence, kernel_size, device)
        all_vars.append(torch.stack(vars_tuple, dim=1))
        variables = torch.cat(all_vars, dim=0).cpu().numpy()
        return variables

    def estimate_parameters(variables, labels):
        '''
        Logistic regression for binary estimation of sharpness
        '''
        model1 = LogisticRegression()
        model1.fit(variables, labels)
        model2 = tree.DecisionTreeClassifier()
        model2.fit(variables, labels)
        model3 = RandomForestClassifier()
        model3.fit(variables, labels)
    
        return model1, model2, model3


    def infer(self):
        with torch.no_grad():
            total_psnr = {}
            total_ssim = {}
            videos = sorted(os.listdir(self.input_path))
            for v in videos:
                video_psnr = []
                video_ssim = []
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*")))
                gt_frames = sorted(glob.glob(os.path.join(self.GT_path, v, "*")))
               
                if not os.path.exists(self.label_path):
                    variables = self.collate_all_vars(input_frames, kernel_size=11, device= self.device)
                    with open('/home/yangt/ssd1/sy32/code/detector/pickle/LogisticRegression_0.5_11.pkl', 'rb') as f:
                        model1 = pickle.load(f)
                    labels = model1.predict(variables)
                else:
                    labels = np.load(os.path.join(self.label_path, v + ".npy"))

                print(self.label_path, labels)
                PreSIndex_list, SubSIndex_list = self.gene_seq_nsf(labels, n_seq=self.n_seq)
             
                input_seqs, new_input_frames = self.gene_seq(input_frames, n_seq=self.n_seq)
                gt_seqs, _ = self.gene_seq(gt_frames, n_seq=self.n_seq)

      
                for in_seq, gt_seq, pre_seq, sub_seq in zip(input_seqs, gt_seqs, PreSIndex_list, SubSIndex_list):   #, bm_seq, label_seq   , bm_seqs, label_seqs
                    start_time = time.time()
                    filename = os.path.basename(in_seq[self.n_seq // 2]).split('.')[0]
                    in_seq.append(new_input_frames[pre_seq[0]])
                    in_seq.append(new_input_frames[sub_seq[self.n_seq-1]])
                    filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in in_seq]
                    def extract_frame_number(filename):
                        return int(filename.split('.')[-1])

                    frame_numbers = [extract_frame_number(name) for name in filenames]

                    inputs = [imageio.imread(p) for p in in_seq]
                    gt = imageio.imread(gt_seq[self.n_seq // 2])
                    
                    h, w, c = inputs[self.n_seq // 2].shape
                    new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                    inputs = [im[:new_h, :new_w, :] for im in inputs]
                    # bms = [im[:new_h, :new_w, np.newaxis] for im in bms]
                    gt = gt[:new_h, :new_w, :]
                    
                    if abs(frame_numbers[2] - frame_numbers[3]) > 7:
                        inputs[-2] = np.zeros_like(inputs[-2])  
                    if abs(frame_numbers[2] - frame_numbers[4]) > 7:
                        inputs[-1] = np.zeros_like(inputs[-1])  

                    in_tensor = self.numpy2tensor(inputs).to(self.device)
                    preprocess_time = time.time()
                   
                    if self.GPUs ==1:
                        output= self.net(in_tensor)  #, bm_tensor, label_tensor
                    else:
                        output = self.forward_chop(in_tensor)
                    forward_time = time.time()
                    output_img = self.tensor2numpy(output)

                    psnr, ssim = self.get_PSNR_SSIM(output_img, gt)
                    video_psnr.append(psnr)
                    video_ssim.append(ssim)
                    total_psnr[v] = video_psnr
                    total_ssim[v] = video_ssim

                    if self.save_image:
                        if not os.path.exists(os.path.join(self.result_path, v)):
                            os.mkdir(os.path.join(self.result_path, v))
                        imageio.imwrite(os.path.join(self.result_path, v, '{}.png'.format(filename)), output_img)
                    postprocess_time = time.time()

                    self.logger.write_log(
                        '> {}-{} PSNR={:.5}, SSIM={:.4} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, filename, psnr, ssim,
                                    preprocess_time - start_time,
                                    forward_time - preprocess_time,
                                    postprocess_time - forward_time,
                                    postprocess_time - start_time))

            sum_psnr = 0.
            sum_ssim = 0.
            n_img = 0
            for k in total_psnr.keys():
                self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(
                    k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k])))
                sum_psnr += sum(total_psnr[k])
                sum_ssim += sum(total_ssim[k])
                n_img += len(total_psnr[k])
            self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum_psnr / n_img, sum_ssim / n_img))

    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[1:1+half]
            img_list_temp.reverse()
            img_list_temp.extend(img_list)
            end_list = img_list[-half - 1:-1]
            end_list.reverse()
            img_list_temp.extend(end_list)
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list, img_list

    def gene_seq_nsf(self, img_list, n_seq):
        img_list = img_list.squeeze().tolist()
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[1:1+half]
            img_list_temp.reverse()
            img_list_temp.extend(img_list)
            end_list = img_list[-half - 1:-1]
            end_list.reverse()
            img_list_temp.extend(end_list)
            img_list = img_list_temp
        # print(img_list)
        PreSIndex_list, SubSIndex_list = self.return_BlurryIndices(img_list)
        preindex = []
        subindex = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            preindex.append(PreSIndex_list[i:i + n_seq])
            subindex.append(SubSIndex_list[i:i + n_seq])
        return preindex, subindex

    def numpy2tensor(self, input_seq, rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = torch.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    def forward_chop(self, *args, shave_h=20, shave_w=20, min_size=160000):
        scale = 1  
        n_GPUs = min(torch.cuda.device_count(), 4)
        args = [a.squeeze().unsqueeze(0) for a in args]

        h, w = args[0].size()[-2:]


        top = slice(0, h//2 + shave_h)
        bottom = slice(h - h//2 - shave_w, h)
        left = slice(0, w//2 + shave_h)
        right = slice(w - w//2 - shave_w, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 6 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]

                y = P.data_parallel(self.net.module, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:

            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave_h=shave_h, shave_w=shave_w, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1:
            y = y[0]

        return y


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
    parser.add_argument('--result_path', type=str, default='../infer_results',
                        help='the path of deblur result')
    args = parser.parse_args()

    if args.default_data == 'REDS':
        args.data_path = './data/deblur/REDS_8x_Random/test'
        args.model_path = '../experiment/model/model_best.pt'
        args.result_path = '../infer_results/bsdtest_reds'
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
    elif args.default_data == 'GOPRO':
        args.data_path = '/home/yangt/nas/BSDtest_all/BSD_3ms24ms/'
        args.model_path = '/home/yangt/ssd1/speinet/experiment/gpa/model/model_best.pt'
        args.result_path = '/home/yangt/nas/speinet/traingopro_testbsd_3ms24ms'
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
    elif args.default_data == 'BSD':
        args.data_path = './data/deblur/BSDtest'
        args.model_path = './model/model_best.pt'
        args.result_path = '../infer_results/BSDtest_finetune'
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
    elif args.default_data == 'BSDtest_all':
        args.data_path = './data/deblur/BSDtest_all/BSD_3ms24ms'
        args.model_path = './model/model_best.pt'
        args.result_path = '../infer_results/BSD_1ms8ms'
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
        args.n_GPUs = 1

    Infer = Inference(args)
    Infer.infer()
