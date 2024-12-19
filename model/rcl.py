import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'

def create_blur_kernel(kernel_size=5):
    kernel = torch.ones((1, 1, kernel_size, kernel_size)).to(torch.float32) / (kernel_size ** 2)
    return kernel

def r_l_per_channel(image_tensor, blur_kernel, num_iterations=1, regularization_strength=0.01):
    deblurred_tensors = []
    kernel_size = blur_kernel.shape[-1]
    padding = int(kernel_size // 2)

    for channel_idx in range(image_tensor.size(1)):
        channel_tensor = image_tensor[:, channel_idx:channel_idx+1, :, :]
        deblurred_channel = channel_tensor.clone().cuda()
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        for iteration in range(num_iterations):
            # Use the blur kernel corresponding to the current channel
            blurred_channel = F.conv2d(deblurred_channel, blur_kernel, padding=padding)

            # Calculate correction factor
            # print('iteration', iteration)
            correction_factor = channel_tensor / blurred_channel
            correction_factor[correction_factor != correction_factor] = 0.0
            correction_factor[correction_factor < 0] = 0.0

            # Laplacian smoothing
            regularized_deblurred_channel = deblurred_channel + regularization_strength * F.conv2d(deblurred_channel, laplacian_kernel, padding=1)

            # Update deblurring channel
            deblurred_channel = correction_factor * regularized_deblurred_channel

        deblurred_tensors.append(deblurred_channel)

    deblurred_tensor = torch.cat(deblurred_tensors, dim=1)
    return deblurred_tensor
