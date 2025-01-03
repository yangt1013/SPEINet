import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import torch.nn as nn
from typing import Optional
from scipy.ndimage.filters import convolve
from scipy.fftpack import fft2, ifft2
from scipy.sparse import spdiags, csr_matrix
# from scipy.sparse.linalg import spsolve
from pypardiso import spsolve
import scipy
import glob
from scipy.signal import gaussian, convolve2d
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
#-------------------------------------------------------------------------------------------------------------
# Sobel Filter
def sobel_filter(tensor):
    # 定义 Sobel 核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().to(tensor.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().to(tensor.device).unsqueeze(0).unsqueeze(0)
    
    # 确保输入为 (batch_size, channels, height, width)
    batch_size, channels, height, width = tensor.size()
    
    # 对每个通道分别进行卷积操作
    grad_x = torch.zeros_like(tensor)
    grad_y = torch.zeros_like(tensor)
    for c in range(channels):
        grad_x[:, c:c+1, :, :] = F.conv2d(tensor[:, c:c+1, :, :], sobel_x, padding=1)
        grad_y[:, c:c+1, :, :] = F.conv2d(tensor[:, c:c+1, :, :], sobel_y, padding=1)
    
    # 计算梯度幅值
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    return grad

#-------------------------------------------------------------------------------------------------------------
# Laplacian Filter
def laplacian_filter(tensor):
    """
    Compute the Laplacian filter on a batch of grayscale or multi-channel images.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
    
    Returns:
        torch.Tensor: Output tensor after applying the Laplacian filter, same shape as input.
    """
    # Define Laplacian kernel
    laplacian_kernel = torch.tensor(
        [[1, 1, 1],
         [1, -8, 1],
         [1, 1, 1]],
        dtype=torch.float32
    ).to(tensor.device).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 3, 3]

    # Ensure input is (batch_size, channels, height, width)
    batch_size, channels, height, width = tensor.size()

    # Initialize output tensor
    filtered = torch.zeros_like(tensor)

    # Apply convolution for each channel independently
    for c in range(channels):
        filtered[:, c:c+1, :, :] = F.conv2d(tensor[:, c:c+1, :, :], laplacian_kernel, padding=1)

    return filtered

#-------------------------------------------------------------------------------------------------------------
# L0 Gradient Minimization

class L0Smoothing:
    """Docstring for L0Smoothing."""

    def __init__(self, img_tensor, param_lambda: Optional[float] = 2e-2, param_kappa: Optional[float] = 2.0):
        """Initialization of parameters"""
        self._lambda = param_lambda
        self._kappa = param_kappa
        self._img_tensor = img_tensor
        self._beta_max = 1e5

    def _circshift(self, psf: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Circular shift for PyTorch Tensors.

        Args:
            psf (torch.Tensor): Input PSF (H, W) or (B, C, H, W).
            shift (torch.Tensor): Shifts corresponding to each dimension.

        Returns:
            torch.Tensor: Circularly shifted PSF.
        """
        shift = shift.int()
        for i in range(shift.size(0)):  # Apply shift along each dimension
            psf = torch.roll(psf, shifts=int(shift[i].item()), dims=i)
        return psf

    def psf2otf(self, psf: torch.Tensor, out_size: tuple, show_plot: Optional[bool] = False) -> torch.Tensor:
        """
        Convert point spread function (PSF) to optical transfer function (OTF) for PyTorch Tensors.

        Args:
            psf (torch.Tensor): Point spread function (H, W) or (B, C, H, W).
            out_size (tuple): Target output size (H, W).
            show_plot (Optional[bool]): If True, visualizes PSF and OTF. Defaults to False.

        Returns:
            torch.Tensor: OTF in Fourier space.
        """
        if not torch.any(psf):
            raise ValueError("Input PSF should not contain zeros.")

        psf_size = torch.tensor(psf.shape[-2:], dtype=torch.int32)  # Handle H, W
        new_psf = torch.zeros(*out_size, dtype=torch.float32, device=psf.device)
        new_psf[:psf_size[0], :psf_size[1]] = psf[:psf_size[0], :psf_size[1]]

        # Circularly shift PSF to center
        shift = -torch.floor(psf_size.float() / 2)
        new_psf = self._circshift(new_psf, shift)

        # Convert to Fourier space (2D FFT)
        otf = torch.fft.fftn(new_psf, dim=(-2, -1))

        return otf.type(torch.complex64)

    def run(self):
        """L0 smoothing implementation"""

        # Normalize the input tensor
        S = self._img_tensor / 256.0
        if S.ndim < 3:
            S = S.unsqueeze(0)  # Add channel dimension if missing

        B, C, H, W = S.shape
        beta = 2 * self._lambda

        # Convert PSF arrays to PyTorch tensors
        psf_x = torch.tensor([[-1, 1]], dtype=torch.float32, device=S.device)
        out_size = (H, W)
        otfx = self.psf2otf(psf_x, out_size)

        psf_y = torch.tensor([[-1], [1]], dtype=torch.float32, device=S.device)
        otfy = self.psf2otf(psf_y, out_size)

        Normin1 = torch.fft.fft2(S, dim=(-2, -1))
        Denormin2 = torch.abs(otfx)**2 + torch.abs(otfy)**2
        if C > 1:
            Denormin2 = Denormin2.unsqueeze(0).repeat(C, 1, 1)

        while beta < self._beta_max:
            Denormin = 1 + beta * Denormin2

            h = torch.diff(S, dim=-1)
            last_col = S[..., 0:1] - S[..., -1:]
            h = torch.cat([h, last_col], dim=-1)

            v = torch.diff(S, dim=-2)
            last_row = S[..., :1, :] - S[..., -1:, :]
            v = torch.cat([v, last_row], dim=-2)

            grad = h**2 + v**2
            if C > 1:
                grad = grad.sum(dim=1, keepdim=True)
            
            # Create mask and broadcast it to match h/v shape
            idx = grad < (self._lambda / beta)
            idx = idx.expand_as(h)

            h[idx] = 0
            v[idx] = 0

            h_diff = -torch.diff(h, dim=-1)
            first_col = h[..., -1:] - h[..., :1]
            h_diff = torch.cat([first_col, h_diff], dim=-1)

            v_diff = -torch.diff(v, dim=-2)
            first_row = v[..., -1:, :] - v[..., :1, :]
            v_diff = torch.cat([first_row, v_diff], dim=-2)

            Normin2 = h_diff + v_diff
            Normin2 = beta * torch.fft.fft2(Normin2, dim=(-2, -1))

            FS = (Normin1 + Normin2) / Denormin
            S = torch.real(torch.fft.ifft2(FS, dim=(-2, -1)))

            beta *= self._kappa

        return S
#paper: Image Smoothing via L0 Gradient Minimization
#github: https://github.com/nrupatunga/L0-Smoothing
#-------------------------------------------------------------------------------------------------------------

# Relative Total Variation (RTV)
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import spsolve
import cv2

def tsmooth(I, lambda_=0.01, sigma=3.0, sharpness=0.02, maxIter=4):
    """
    Smooth the input tensor.

    Args:
        I: Input tensor of shape (B, C, H, W).
        lambda_: Smoothing regularization weight.
        sigma: Standard deviation for low-pass filter.
        sharpness: Sharpness control parameter.
        maxIter: Number of iterations.

    Returns:
        Smoothed tensor of shape (B, C, H, W) as torch.float32.
    """
    if not isinstance(I, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    if I.dim() != 4:
        raise ValueError(f"Input `I` must have shape (B, C, H, W), but got shape {I.shape}.")

    B, C, H, W = I.shape
    x = I.clone()
    sigma_iter = sigma
    lambda_ /= 2.0
    dec = 2.0

    for _ in range(maxIter):
        wx, wy = computeTextureWeights(x, sigma_iter, sharpness)
        x = solveLinearEquation(I, wx, wy, lambda_)
        sigma_iter = max(sigma_iter / dec, 0.5)

    return x  

def computeTextureWeights(fin, sigma, sharpness):
    """
    Compute texture weights for input tensor.

    Args:
        fin (torch.Tensor): Tensor of shape (B, C, H, W).
        sigma (float): Standard deviation for Gaussian filtering.
        sharpness (float): Sharpness control parameter.

    Returns:
        wx, wy: Texture weights of shape (B, C, H, W).
    """
    if fin.dim() != 4:
        raise ValueError(f"Input `fin` must have shape (B, C, H, W), but got shape {fin.shape}.")

    B, C, H, W = fin.shape
    device = fin.device
    vareps_s = torch.tensor(sharpness, device=device)
    vareps = torch.tensor(0.001, device=device)

    # Compute horizontal and vertical gradients
    fx = fin[:, :, :, 1:] - fin[:, :, :, :-1]  # Horizontal gradient
    fx = F.pad(fx, (0, 1, 0, 0), mode="replicate")  # Pad along width

    fy = fin[:, :, 1:, :] - fin[:, :, :-1, :]  # Vertical gradient
    fy = F.pad(fy, (0, 0, 0, 1), mode="replicate")  # Pad along height

    # Compute magnitude of gradients
    magnitude = torch.sqrt(fx**2 + fy**2)
    wto = torch.maximum(torch.mean(magnitude, dim=(2, 3), keepdim=True), vareps_s)

    # Apply low-pass filter
    fbin = lpfilter(fin, sigma)

    gfx = fbin[:, :, :, 1:] - fbin[:, :, :, :-1]
    gfx = F.pad(gfx, (0, 1, 0, 0), mode="replicate")

    gfy = fbin[:, :, 1:, :] - fbin[:, :, :-1, :]
    gfy = F.pad(gfy, (0, 0, 0, 1), mode="replicate")

    # Compute texture weights
    wtbx = torch.maximum(torch.mean(torch.abs(gfx), dim=(2, 3), keepdim=True), vareps)
    wtby = torch.maximum(torch.mean(torch.abs(gfy), dim=(2, 3), keepdim=True), vareps)

    wx = wtbx * wto
    wy = wtby * wto

    return wx, wy

def lpfilter(FImg, sigma):
    """
    Apply low-pass filter to each channel of the input tensor.

    Args:
        FImg (torch.Tensor): Tensor of shape (B, C, H, W).
        sigma (float): Standard deviation for Gaussian filtering.

    Returns:
        Filtered tensor of shape (B, C, H, W).
    """
    B, C, H, W = FImg.shape
    FBImg = torch.zeros_like(FImg)
    for b in range(B):
        for c in range(C):
            channel = FImg[b, c, :, :].cpu().numpy()
            FBImg[b, c, :, :] = torch.from_numpy(conv2_sep(channel, sigma)).to(FImg.device)
    return FBImg

def conv2_sep(im, sigma):
    """
    Apply separable convolution with Gaussian kernel.

    Args:
        im (numpy.ndarray): 2D input array.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        Filtered 2D array.
    """
    ksize = max(1, round(5 * sigma))
    if ksize % 2 == 0:
        ksize += 1
    g = cv2.getGaussianKernel(ksize, sigma)
    ret = cv2.filter2D(im, -1, g)
    ret = cv2.filter2D(ret, -1, g.T)
    return ret

def solveLinearEquation(IN, wx, wy, lambda_):
    """
    Solve the linear equation for smoothed output.

    Args:
        IN (torch.Tensor): Input tensor of shape (B, C, H, W).
        wx, wy (torch.Tensor): Texture weights of shape (B, C, H, W).
        lambda_ (float): Regularization weight.

    Returns:
        Smoothed tensor of shape (B, C, H, W).
    """
    B, C, H, W = IN.shape
    OUT = torch.zeros_like(IN)

    for b in range(B):
        for c in range(C):
            tin = IN[b, c, :, :].view(-1).cpu().numpy()
            wx_flat = wx[b, c, :, :].view(-1).cpu().numpy()
            wy_flat = wy[b, c, :, :].view(-1).cpu().numpy()

            dx = -lambda_ * wx_flat
            dy = -lambda_ * wy_flat

            k = H * W
            B = np.vstack((dx, dy))
            d = [-H, -1]
            A = spdiags(B, d, k, k)

            e = dx
            w = np.pad(dx[:-H], (H, 0), "constant")
            s = dy
            n = np.pad(dy[:-1], (1, 0), "constant")
            D = 1 - (e + w + s + n)
            A = A + A.T + spdiags(D, 0, k, k)

            A = csr_matrix(A)

            tout = spsolve(A.astype(np.float64), tin.astype(np.float64))
            OUT[b, c, :, :] = torch.from_numpy(tout.reshape((H, W))).to(IN.device)

    return OUT

#paper:Structure Extraction from Texture via Relative Total Variation
#github:https://github.com/guchengxi1994/RTV-in-Python
#------------------------------------------------------------------------------------------------------------------
# Wiener Filter
def gaussian_kernel(kernel_size=3):
    """
    生成一个二维高斯核。
    """
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.T)
    h /= np.sum(h)
    return h

def wiener_filter(img, kernel, K):
    """
    Perform Wiener filtering on the input image.

    Args:
        img (torch.Tensor): Input image tensor of shape (B, C, H, W).
        kernel (numpy.ndarray or torch.Tensor): Convolution kernel.
        K (float): Noise-to-signal power ratio.

    Returns:
        torch.Tensor: Filtered image tensor.
    """
    # Ensure kernel is a PyTorch tensor
    if isinstance(kernel, np.ndarray):
        kernel = torch.from_numpy(kernel).float()
    
    # Ensure the kernel is on the same device as the image
    kernel = kernel.to(img.device)

    # Normalize the kernel
    kernel /= kernel.sum()

    # Pad the kernel to match the size of the input image
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    kernel = F.pad(kernel, [
        0, img.shape[-1] - kernel.shape[-1],  # Pad width
        0, img.shape[-2] - kernel.shape[-2]   # Pad height
    ])
    kernel = kernel.expand(img.shape[0], img.shape[1], -1, -1)  # Match batch and channel dimensions

    # Perform FFT-based Wiener filtering
    img_fft = torch.fft.fftn(img, dim=(-2, -1))  # FFT of the input image
    kernel_fft = torch.fft.fftn(kernel, dim=(-2, -1))  # FFT of the kernel

    kernel_fft_conj = torch.conj(kernel_fft)  # Conjugate of the kernel FFT
    wiener_kernel = kernel_fft_conj / (torch.abs(kernel_fft) ** 2 + K)  # Wiener filter kernel

    filtered_fft = img_fft * wiener_kernel  # Apply Wiener filter in frequency domain
    filtered_img = torch.real(torch.fft.ifftn(filtered_fft, dim=(-2, -1)))  # Inverse FFT to get filtered image

    return filtered_img


#paper:Robinson, E. A., & Treitel, S. (1967). Principles of digital Wiener filtering. Geophysical Prospecting, 15(3), 311-332.
#github:https://github.com/tranleanh/wiener-median-comparison
#---------------------------------------------------------------------------------------------------------------------
# Richardson-Lucy Deconvolution

class RL_Deconv(nn.Module):
    def __init__(self, kernel=5, sigma=1.0, channels=3, iterations=5, tolerance=1e-5):
        super(RL_Deconv, self).__init__()
        if not isinstance(kernel, int):
            raise ValueError(f"Kernel size must be an integer, but got {type(kernel).__name__}: {kernel}")
        
        self.channels = channels
        self.iterations = iterations
        self.tolerance = tolerance
        self.pad_size = kernel // 2

        # Initialize the kernel
        kernel_tmp = np.zeros((kernel, kernel), dtype=np.float32)
        divisor = -2.0 * sigma * sigma
        norm_coef = 0
        for i in range(-self.pad_size, self.pad_size + 1):
            for j in range(-self.pad_size, self.pad_size + 1):
                value = np.exp((i * i + j * j) / divisor)
                kernel_tmp[i + self.pad_size, j + self.pad_size] = value
                norm_coef += value

        kernel_tmp = kernel_tmp / norm_coef  # Normalize the kernel

        # Create the kernel with the correct shape
        self.kernel = torch.tensor(kernel_tmp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        self.kernel = self.kernel.repeat(channels, 1, 1, 1)  # Shape: [channels, 1, H, W]
        self.kernel_flipped = torch.flip(self.kernel, dims=(2, 3))  # Flip kernel for error estimation

    def forward(self, inputs):
        device = inputs.device  # Get the device of the input tensor
        latent_est = inputs
        paddings = (self.pad_size, self.pad_size, self.pad_size, self.pad_size)  # (left, right, top, bottom)

        # Move kernel tensors to the same device as the input
        kernel = self.kernel.to(device)
        kernel_flipped = self.kernel_flipped.to(device)

        for i in range(self.iterations):
            # Pad latent estimate for convolution
            latent_est_pad = F.pad(latent_est, paddings, mode='reflect')
            
            # Perform convolution
            est_conv = F.conv2d(latent_est_pad, kernel, groups=self.channels)
            
            # Calculate relative blur
            relative_blur = inputs / (est_conv + 1e-8)  # Avoid division by zero
            
            # Pad relative blur for convolution
            relative_blur_pad = F.pad(relative_blur, paddings, mode='reflect')
            
            # Error estimation
            error_est = F.conv2d(relative_blur_pad, kernel_flipped, groups=self.channels)
            
            # Update latent estimate
            latent_est = latent_est * error_est
            
            #Check for convergence
            # if torch.abs(1 - torch.mean(error_est)) < self.tolerance:
            #     print(f"Converged at iteration: {i}")
            #     break

        return latent_est
#paper:  Fish, D. A., Brinicombe, A. M., Pike, E. R., & Walker, J. G. (1995). Blind deconvolution by means of the Richardson–Lucy algorithm. JOSA A, 12(1), 58-65.
#github:https://github.com/Pol22/Richardson-Lucy
#-------------------------------------------------------------------------------------------------------------------
# Total Variation Minimization

def zero_pad(image, shape, position='corner'):
    """
    Extend a 2D image to a certain size with zeros.

    Args:
        image (numpy.ndarray): Input 2D array of shape (H, W).
        shape (tuple): Target output shape (H, W).
        position (str): Position of the input image in the output ('corner' or 'center').

    Returns:
        numpy.ndarray: Zero-padded image.
    """
    if len(shape) != 2 or len(image.shape) != 2:
        raise ValueError(f"Expected 2D input and shape, but got input shape {image.shape} and target shape {shape}.")

    imshape = np.array(image.shape)
    shape = np.array(shape)

    if np.any(shape < imshape):
        raise ValueError("Target shape must be larger than or equal to the input shape.")

    dshape = shape - imshape
    pad_img = np.zeros(shape, dtype=image.dtype)

    if position == 'center':
        offsets = dshape // 2
    else:
        offsets = np.zeros_like(dshape)

    slices = tuple(slice(offsets[i], offsets[i] + imshape[i]) for i in range(len(shape)))
    pad_img[slices] = image

    return pad_img

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Args:
        psf (numpy.ndarray): PSF array (2D).
        shape (tuple): Target output shape (H, W).

    Returns:
        numpy.ndarray: OTF array.
    """
    # 确保目标形状是二维
    if len(shape) > 2:
        shape = shape[-2:]  # 提取 (H, W)
    
    if psf.ndim != 2:
        raise ValueError(f"Expected 2D input for PSF, but got shape {psf.shape}.")

    if psf.shape != shape:
        psf_pad = zero_pad(psf, shape, position='corner')  # Zero padding
    else:
        psf_pad = psf

    # Circularly shift OTF
    for axis, axis_size in enumerate(psf.shape):
        psf_pad = np.roll(psf_pad, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf_pad)

    return np.real_if_close(otf, tol=1e-5)
 
def Dive(X, Y):
    """
    Transpose of the forward finite difference operator.

    Args:
        X, Y (numpy.ndarray): Input arrays for differences.

    Returns:
        numpy.ndarray: Divergence result.
    """
    # Handle case where Y has only one row
    if Y.shape[0] < 2:
        Y_padded = np.vstack([Y, np.zeros_like(Y)])  # Add a zero row
    else:
        Y_padded = Y

    fwd_diff_rowX = np.expand_dims(X[:, -1] - X[:, 1], axis=1)
    DtXY = np.concatenate((fwd_diff_rowX, -np.diff(X, axis=1)), axis=1)

    fwd_diff_rowY = np.expand_dims(Y_padded[-1, :] - Y_padded[1, :], axis=0)
    DtXY = DtXY + np.concatenate((fwd_diff_rowY, -np.diff(Y_padded, axis=0)), axis=0)

    return DtXY


def getC(image, kernel):
    """
    Compute components for deconvolution.

    Args:
        image (numpy.ndarray): Input image (H, W).
        kernel (numpy.ndarray): Convolution kernel (H, W).

    Returns:
        tuple: (eigsK, KtF, eigsDtD, eigsKtK)
    """
    sizeF = image.shape[-2:]  # 获取 (H, W)

    # 确保 kernel 是二维
    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D kernel, but got {kernel.ndim}D kernel with shape {kernel.shape}.")

    eigsK = psf2otf(kernel, sizeF)   # Compute eigenvalues
    KtF = np.real(np.fft.ifft2(np.conj(eigsK) * np.fft.fft2(image)))  # K^T * F

    # 修复 diff_kernelX 和 diff_kernelY 的维度
    diff_kernelX = np.array([[1, -1]])  # Shape: (1, 2)
    diff_kernelY = np.array([[1], [-1]])  # Shape: (2, 1)

    eigsDtD = np.abs(psf2otf(diff_kernelX, sizeF))**2 + np.abs(psf2otf(diff_kernelY, sizeF))**2  # D^T * D
    eigsKtK = np.abs(eigsK)**2  # K^T * K

    return eigsK, KtF, eigsDtD, eigsKtK


def ForwardD(U):
    # % Forward finite difference operator
    end_col_diff = np.expand_dims((U[:,0]- U[:,-1]),axis=1)
    end_row_diff = np.expand_dims((U[0,:] - U[-1,:]),axis=0)
    Dux = np.concatenate((np.diff(U,1,1), end_col_diff),axis=1)     # discrete gradient operators
    Duy = np.concatenate((np.diff(U,1,0), end_row_diff),axis=0)
    return (Dux,Duy)

def fval(D1X,D2X,eigsK,X,img,mu):
    f = np.sum(np.sum(np.sqrt(D1X**2 + D2X**2)))   
    KXF = np.real(np.fft.ifft2(eigsK * np.fft.fft2(X))) - img
    f = f + mu/2 * np.linalg.norm(KXF,'fro')**2
    return f

def ftvd(kernel, img, beta=10, gamma=1.618, max_itr=500, relchg=1e-3, mu=500):
    """
    Alternating Directions Method (ADM) applied to TV/L2.
    """
    Lam1 = np.zeros_like(img.cpu().numpy())
    Lam2 = np.zeros_like(img.cpu().numpy())
    eigsK, KtF, eigsDtD, eigsKtK = getC(img.cpu().numpy(), kernel)

    X = img.cpu().numpy().copy()
    D1X, D2X = ForwardD(X)

    for ii in range(max_itr):
        # Shrinkage step
        Z1 = D1X + Lam1 / beta
        Z2 = D2X + Lam2 / beta
        V = np.sqrt(Z1**2 + Z2**2)
        V[V == 0] = 1
        V = np.maximum(V - 1 / beta, 0) / V
        Y1 = Z1 * V
        Y2 = Z2 * V

        # X subproblem
        Xp = X.copy()
        X = (mu * KtF - Dive(Lam1, Lam2)) / beta + Dive(Y1, Y2)
        X = np.fft.fft2(X) / (eigsDtD + (mu / beta) * eigsKtK)
        X = np.real(np.fft.ifft2(X))

        # 对齐批次大小
        if Xp.shape[0] != X.shape[0]:
            min_batch = min(Xp.shape[0], X.shape[0])
            Xp = Xp[:min_batch]
            X = X[:min_batch]

        # 展平并计算相对变化
        Xp_flat = Xp.reshape(-1, Xp.shape[-2] * Xp.shape[-1])
        X_flat = X.reshape(-1, X.shape[-2] * X.shape[-1])
        relchg_iter = np.linalg.norm(Xp_flat - X_flat, 'fro') / np.linalg.norm(Xp_flat, 'fro')

        # Check for convergence
        if relchg_iter < relchg:
            break

        # Update multipliers
        D1X, D2X = ForwardD(X)
        Lam1 = Lam1 - gamma * beta * (Y1 - D1X)
        Lam2 = Lam2 - gamma * beta * (Y2 - D2X)

    return torch.tensor(X, device=img.device), ii + 1



#paper:Wang, Yilun, et al. "A new alternating minimization algorithm for total variation image reconstruction." SIAM Journal on Imaging Sciences 1.3 (2008): 248-272.
#github:https://github.com/JoshuaEbenezer/ftvd
