"""
Common functions 
"""

import torch
import numpy as np
import scipy
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims, source_spacing, receiver_spacing,
             source_depth, receiver_depth):
    """
        Create arrays containing the source and receiver locations
        Args:
            num_shots: nunmber of shots
            num_sources_per_shot: number of sources per shot
            num_receivers_per_shot： number of receivers per shot
            num_dims: dimension of velocity model
        return:
            x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
            x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
    """
    x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    if source_depth != 0:
        x_s[:, 0, 0] = source_depth
    x_s[:, 0, 1] = torch.arange(1, num_shots + 1).float() * source_spacing
    x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    if receiver_depth != 0:
        x_r[:, :, 0] = receiver_depth
    x_r[0, :, 1] = torch.arange(1, num_receivers_per_shot + 1).float() * receiver_spacing
    x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

    return x_s, x_r

def loadtruemodel(data_dir, num_dims, vmodel_dim):
    """
        Load the true model
    """

    if num_dims != len(vmodel_dim.reshape(-1)):
        raise Exception('Please check the size of model_true!!')
    # prefer the depth direction first, that is the shape is `[nz, (ny, (nx))]`
    if num_dims == 2:
        model_true = (np.fromfile(data_dir, np.float32).reshape(vmodel_dim[1], vmodel_dim[0]))
        model_true = np.transpose(model_true, (1, 0))  # I prefer having depth direction first
    else:

        raise Exception('Please check the size of model_true!!')

    model_true = torch.Tensor(model_true)  # Convert to a PyTorch Tensor

    return model_true

def ComputeSNR(rec, target):
    """
       Calculate the SNR between reconstructed image and true  image
    """
    if torch.is_tensor(rec):
        rec = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()

    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')

    snr = 0.0
    if len(rec.shape) == 3:
        for i in range(rec.shape[0]):
            rec_ind = rec[i, :, :].reshape(np.size(rec[i, :, :]))
            target_ind = target[i, :, :].reshape(np.size(rec_ind))
            s = 10 * np.log10(sum(target_ind ** 2) / sum((rec_ind - target_ind) ** 2))
            snr = snr + s
        snr = snr / rec.shape[0]
    elif len(rec.shape) == 2:
        rec = rec.reshape(np.size(rec))
        target = target.reshape(np.size(rec))
        snr = 10 * np.log10(sum(target ** 2) / sum((rec - target) ** 2))
    else:
        raise Exception('Please reshape the Rec to correct Dimension!!')
    return snr

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L = 255
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    """
    gaussian filter
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    create the window for computing the SSIM
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ComputeSSIM(img1, img2, window_size=11, size_average=True):
    """
    compute the SSIM between img1 and img2
    """
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))

    if len(img1.size()) == 2:
        d = img1.size()
        img1 = img1.view(1, 1, d[0], d[1])
        img2 = img2.view(1, 1, d[0], d[1])
    elif len(img1.size()) == 3:
        d = img1.size()
        img1 = img1.view(d[2], 1, d[0], d[1])
        img2 = img2.view(d[2], 1, d[0], d[1])
    else:
        raise Exception('The shape of image is wrong!!!')
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def ComputeRE(rec, target):
    """
    Compute relative error between the rec and target
    """
    if torch.is_tensor(rec):
        rec = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()

    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')

    rec = rec.reshape(np.size(rec))
    target = target.reshape(np.size(rec))
    rerror = np.sqrt(sum((target - rec) ** 2)) / np.sqrt(sum(target ** 2))

    return rerror