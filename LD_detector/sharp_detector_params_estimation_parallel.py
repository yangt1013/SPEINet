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
    parser.add_argument('--dir-path', default='/home/yangt/ssd1/dataset/GOPRO_Large_all/train', type=str, help='path to the clean frames directory')
    parser.add_argument('--device','-d', default='cuda', type=str, help='device')
    parser.add_argument('--window-range', '-w', type=tuple, default=(1,15), help='Range of possible window sizes.')
    parser.add_argument('--threshold', '-t', type=float, default=5, help='Threshold for sharpness definition. All the frames                                                  deduced by averaging not greater than 5 frames is considered sharp')
    parser.add_argument('--ratio','-r', default=0.5,type=float, help='Ratio of sharp frames in the generated blurry sequence') #default=0.5,
    parser.add_argument('--kernel-size','-k', default=11, type=int, help='The kernel size of the detector.')#default=11, 
    parser.add_argument('--seed', '-s', default = 4000, type=int, help='Set the seed for reproducibility of the results.')#1234 51%  #2000 47%  #4000  50%
    return parser.parse_args()

def check_arguments(windows, threshold, ratio):
    assert threshold in range(*windows), 'Threshold should be in the range of windows'
    if ratio is None:
       raise ValueError("Ratio should not be None")
    assert 0 <= ratio <= 1, 'Ratio should be in the range of [0,1]'

def load_single_sequence(dir_path, p):
    list_images = sorted(os.listdir(dir_path))
    list_fname_path = [os.path.join(dir_path, frame) for frame in list_images]
    out = p.map(imageio.imread, list_fname_path)
    return out
    
def generate_blurry_sequence(frames: list, window_range: tuple[int], ratio: float, threshold: int, seed: Optional[int] = None):
    blurry_frames = []
    gt_frames = []
    frames_labels = []
    if seed is not None:
        random.seed(seed)

    while len(frames) != 0:
        random_label = torch.tensor(int((random.random() < ratio) or (len(frames) <= threshold)))
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
    blurry_frames = torch.stack(blurry_frames).permute(0,3,1,2)
    gt_frames = torch.stack(gt_frames).permute(0,3,1,2)
    frames_labels = torch.stack(frames_labels)

    return blurry_frames, frames_labels, gt_frames

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

def find_subdirs_with_name_sharp(dir_path):
    subdirs = []
    for subdir in sorted(os.listdir(dir_path)):
        subdirs.append(subdir)
    return subdirs

def collate_all_vars(dir_path:str, window_range:Tuple[int], threshold: int, ratio:float, kernel_size: int, p, device, seed: Optional[int]=None):
    subdirs = find_subdirs_with_name_sharp(dir_path)
    all_vars = []
    all_labels = []
    loader  = tqdm(subdirs, desc='Loading videos...')
    for idx, subdir in enumerate(loader):
        frames = load_single_sequence(os.path.join(dir_path, subdir), p)
        blurry_sequence, labels, gt_sequence = generate_blurry_sequence(frames, window_range, ratio, threshold, seed)
        vars_tuple = generate_vars(blurry_sequence, kernel_size, device)#len = 6
        all_vars.append(torch.stack(vars_tuple, dim=1))#这个是正确的 len = 1
        all_labels.extend(labels)
        loader.set_description_str(f'Total frames: {len(all_labels)}, Actual sharp-ratio {np.mean(all_labels):.2f}')
        # if idx >=2:
        #     break
    variables = torch.cat(all_vars, dim=0).cpu().numpy()
    labels = torch.tensor(all_labels).numpy()
    return variables, labels, gt_sequence

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
    args = parse_arguments()
    check_arguments(args.window_range, args.threshold, args.ratio)
    p = Pool(processes=16)
    variables, labels, gt_sequence = collate_all_vars(p=p, **vars(args))

    train_variables, validation_variables, train_labels, validation_labels = train_test_split(variables, labels, test_size=0.1, random_state=args.seed)
    model1, model2, model3 = estimate_parameters(train_variables, train_labels)
    
    with open(f'LogisticRegression_{args.ratio}_{args.kernel_size}.pkl', 'wb') as f:
        pickle.dump(model1, f)
    with open(f'DecisionTree_{args.ratio}_{args.kernel_size}.pkl', 'wb') as f:
        pickle.dump(model2, f)
    with open(f'RandomForest_{args.ratio}_{args.kernel_size}.pkl', 'wb') as f:
        pickle.dump(model3, f)

    tp1, tn1, fp1, fn1 = calculate_metrics(validation_labels, model1.predict(validation_variables))
    print(f'Results for a total amount of {variables.shape[0]} frames')
    coffecients1 = dict(zip(['LAP1','MIS3','WAV1','GRA7','STA3','DCT3'] ,model1.coef_[0]))
    print(f'model1: Coefficients {coffecients1}')   
    print(f'model1: Logistc Intercept {model1.intercept_[0]}')
    print(f'model1: Logistic Accuracy: {(model1.score(validation_variables, validation_labels))*100:.1f}')
    print(f'model1: Logistic Recall: {(recall_score(model1.predict(validation_variables), validation_labels))*100:.1f}')
    print(f'model1: Logistic F1-score: {(f1_score(model1.predict(validation_variables), validation_labels))*100:.1f}')
    print(f'model1: Logistic Precision: {(precision_score(model1.predict(validation_variables), validation_labels))*100:.1f}')
    ###
    tp2, tn2, fp2, fn2 = calculate_metrics(validation_labels, model1.predict(validation_variables))
    print(f'model2: Decision Accuracy: {(model2.score(validation_variables, validation_labels))*100:.1f}')
    print(f'model2: Decision Recall: {(recall_score(model2.predict(validation_variables), validation_labels))*100:.1f}')
    print(f'model2: Decision F1-score: {(f1_score(model2.predict(validation_variables), validation_labels))*100:.1f}')
    print(f'model2: Decision Precision: {(precision_score(model2.predict(validation_variables), validation_labels))*100:.1f}')
    ###
    tp3, tn3, fp3, fn3 = calculate_metrics(validation_labels, model1.predict(validation_variables))
    print(f'model3: Random Accuracy: {(model3.score(validation_variables, validation_labels))*100:.1f}')
    print(f'model3: Random Recall: {(recall_score(model3.predict(validation_variables), validation_labels))*100:.1f}')
    print(f'model3: Random F1-score: {(f1_score(model3.predict(validation_variables), validation_labels))*100:.1f}')
    print(f'model3: Random Precision: {(precision_score(model3.predict(validation_variables), validation_labels))*100:.1f}')
    df = pd.DataFrame({'name': ['Logistic', 'Decision', 'Random'],                  
                        'ratio': [args.ratio, args.ratio, args.ratio],
                        'kernel_size': [args.kernel_size, args.kernel_size,args.kernel_size],
                        'window_range': [args.window_range, args.window_range,args.window_range],
                        'true_positive': [tp1, tp2, tp3],
                        'true_negative': [tn1, tn2, tn3],
                        'false_positive': [fp1, fp2, fp3],
                        'false_negative': [fn1, fn2, fn3],
                        'positive': [tp1+fn1, tp2+fn2, tp3+fn3],
                        'negative': [fp1+tn1, fp2+tn2, fp3+tn3],
                        'predict_positive': [tp1+fp1, tp2+fp2, tp3+fp3],
                        'predict_negative': [tn1+fn1, tn1+fn1, tn1+fn1],
                        'coffecients1': [coffecients1, None, None],})
    df.to_csv('output.csv', mode= 'a', index=False, header=False) 
    print('csv finished')
    print('all finished')
    

    