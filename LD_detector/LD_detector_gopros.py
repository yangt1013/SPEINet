'''
This script is used to estimate the sharpness detector parameters to be used in the TiaNet
'''
from argparse import ArgumentParser
from tqdm import tqdm
import os 
import random
import numpy as np
from typing import Optional, Tuple
from multiprocessing import Pool
from ptwt import wavedec2
import pywt
from torch.nn.functional import lp_pool2d, avg_pool2d, conv2d
import torch
from torch import Tensor
import torchvision
import imageio
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_recall_curve, precision_score, confusion_matrix
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICE'] = '6'
torch.cuda.set_device(6)
import pickle
def parse_arguments():
    parser = ArgumentParser(description='Sharpness Detector Parameters Estimation')
    parser.add_argument('--dir-path', default='/home/yangt/ssd1/speinet_dataset/GoProS/test', type=str, help='path to the clean frames directory')
    parser.add_argument('--device','-d', default='cuda', type=str, help='device') 
    parser.add_argument('--kernel-size','-k', default=11, type=int, help='The kernel size of the detector.')#default=11
    return parser.parse_args()


def load_single_sequence(dir_path, p):
    list_images = sorted([
        os.path.join(dir_path, fname)
        for fname in os.listdir(dir_path)
        if fname.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])
    out = list(p.map(read_image, list_images))  # 加载图像
    # print(f"Loaded image shapes: {[img.shape for img in out]}")  # 打印形状
    return np.array(out)  # 返回 NumPy 数组


def read_image(image_path):
    """
    用于加载单个图像文件
    """
    return imageio.imread(image_path)

    

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
    """
    提取特征的核心函数
    """
    frames = frames.to(device)  # 转移到指定设备
    if frames.ndim == 3:  # 如果是 [B, H, W] 格式
        frames = frames.unsqueeze(1)  # 添加通道维度 -> [B, 1, H, W]
    elif frames.ndim == 4 and frames.shape[1] == 720:
        frames = frames.permute(0, 3, 1, 2)  # 修正形状 -> [B, C, H, W]

    gray_frames = torchvision.transforms.Grayscale()(frames) / 255.0  # 转为灰度图并归一化
    lap = focus_measure_lap1(gray_frames, kernel_size)
    mis3 = focus_measure_mis3(gray_frames, kernel_size)
    wave1 = focus_measure_wave1(gray_frames, kernel_size)
    gra_7 = focus_measure_gra7(gray_frames, kernel_size)
    sta_3 = focus_measure_sta3(gray_frames, kernel_size)
    dct_3 = focus_measure_dct3(gray_frames, kernel_size)
    return lap, mis3, wave1, gra_7, sta_3, dct_3


def find_subdirs_with_name_sharp(dir_path):
    subdirs = []
    for subdir in sorted(os.listdir(dir_path)):
        subdirs.append(subdir)
    return subdirs

def collate_all_vars(dir_path: str, kernel_size: int, p, device):
    """
    加载模糊数据和标签，并提取特征用于训练
    """
    blur_dir = os.path.join(dir_path, "blur")
    label_dir = os.path.join(dir_path, "label")

    # 找到 blur 和 label 目录下的所有子文件夹和对应标签文件
    subdirs = sorted(os.listdir(blur_dir))
    all_vars = []
    all_labels = []
    loader = tqdm(subdirs, desc='Loading data...')

    for subdir in loader:
        # 加载子文件夹中的模糊图像
        blur_folder = os.path.join(blur_dir, subdir)
        label_file = os.path.join(label_dir, f"{subdir}.npy")

        # 检查文件是否存在
        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} does not exist!")
            continue

        # 加载模糊图像
        frames = load_single_sequence(blur_folder, p)

        # 加载对应标签
        labels = np.load(label_file)  # 标签是一个 NumPy 数组
        if len(frames) != len(labels):
            print(f"Error: Mismatch in frame count and label count for {subdir}!")
            continue

        # 将图像转换为 Tensor 并提取特征
        vars_tuple = generate_vars(torch.tensor(frames).float(), kernel_size, device)  # 计算特征

        # 收集特征和标签
        all_vars.append(torch.stack(vars_tuple, dim=1))
        all_labels.extend(labels)

    # 合并所有特征和标签
    variables = torch.cat(all_vars, dim=0).cpu().numpy()  # 转为 NumPy 格式
    labels = np.array(all_labels)  # 转为 NumPy 格式
    return variables, labels


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

    
def calculate_metrics(labels, predict):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(labels)):
        if labels[i] == 1 and predict[i] == 1:
            tp += 1
        elif labels[i] == 0 and predict[i] == 0:
            tn += 1
        elif labels[i] == 0 and predict[i] == 1:
            fp += 1
        elif labels[i] == 1 and predict[i] == 0:
            fn += 1
    return tp, tn, fp, fn


if __name__ == '__main__':
    # 解析参数
    args = parse_arguments()
    p = Pool(processes=16)

    # 加载测试数据
    print("Loading test data...")
    test_variables, test_labels = collate_all_vars(args.dir_path, args.kernel_size, p, args.device)
    p.close()
    p.join()

    # 加载训练好的模型
    print("Loading trained models...")
    model_paths = {
        "LogisticRegression": '/home/yangt/nas/video_deblurring_nas/sy32/code/detector/GoProS_LogisticRegression_11.pkl',
        "DecisionTree": '/home/yangt/nas/video_deblurring_nas/sy32/code/detector/GoProS_DecisionTree_11.pkl',
        "RandomForest": '/home/yangt/nas/video_deblurring_nas/sy32/code/detector/GoProS_RandomForest_11.pkl'
    }
    models = {}
    for name, path in model_paths.items():
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)

    # 使用模型进行预测并计算性能指标
    results = []
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        predictions = model.predict(test_variables)
        accuracy = model.score(test_variables, test_labels) * 100
        recall = recall_score(test_labels, predictions) * 100
        precision = precision_score(test_labels, predictions) * 100
        f1 = f1_score(test_labels, predictions) * 100
        tp, tn, fp, fn = calculate_metrics(test_labels, predictions)
        
        # 打印结果
        print(f"{model_name} - Accuracy: {accuracy:.2f}%, Recall: {recall:.2f}%, Precision: {precision:.2f}%, F1 Score: {f1:.2f}%")
        
        # 保存结果
        results.append({
            'name': model_name,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        })

    # 保存所有结果到 CSV
    print("Saving results to CSV...")
    output_path = '/home/yangt/nas/video_deblurring_nas/sy32/code/detector/gopros_output.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_path, mode='w', index=False, header=True)
    print(f"Results saved to {output_path}")

    