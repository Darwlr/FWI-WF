"""
Loss functions
"""

import torch
import numpy as np


def hilbert(data_in):
  nt, nr = data_in.shape
  transforms = torch.fft.fftn(data_in,dim=0)
  #print(transforms.shape)
  transforms[1:nt//2,:]      *= 2.0
  transforms[nt//2 + 1: nt,:]  = 0+0j
  transforms[0,:] = 0
  data_out = torch.abs(torch.fft.ifftn(transforms,dim=0))
  return data_out


def _fft_loss(dm, do, f=6.0):
    residual = dm - do
    W = torch.diag((np.abs(2 * torch.pi * f)) * torch.ones(dm.shape[0])).cuda()

    residual_hat = hilbert(residual)

    l1_loss = torch.mm(W, torch.mm(residual_hat, residual_hat.T))

    return torch.sum(l1_loss)


def depth_weighted_matrix(m):
    weights = torch.sqrt(torch.arange(m.shape[0], dtype=torch.float32) + 8).view(-1, 1)
    weights = weights.cuda()
    return weights



def FFT_loss(d_m, d_o):
    total_loss = torch.tensor(0.0).cuda()
    for i in range(d_m.shape[1]):
        total_loss += _fft_loss(d_m[:, i, :], d_o[:, i, :])
    return total_loss


def transform(d_sim, d_obs, theta=1.1, trans_type='linear'):
    """
        do the transform for the signal d_sim and d_obs
        Args:
            d_sim, d_obs: seismic data with the shape [num_time_steps,num_shots*num_receivers_per_shot]
            trans_type: type for transform
            theta:the scalar variable for transform
        return:
            output with transfomation
    """
    assert len(d_sim.shape) == 2
    c = 0.0
    device = d_sim.device
    if trans_type == 'linear':
        min_value = torch.min(d_sim.detach().min(), d_obs.detach().min())
        mu, nu = d_sim, d_obs
        c = -min_value if min_value<0 else 0
        c = c * theta       # c = 1.1 x max(), corresponding to c in equation (4)
        d = torch.ones(d_sim.shape).to(device)
    elif trans_type == 'abs':
        mu, nu = torch.abs(d_sim), torch.abs(d_obs)
        d = torch.sign(d_sim).to(device)
    elif trans_type == 'square':
        mu = d_sim * d_sim
        nu = d_obs * d_obs
        d = 2 * d_sim
    elif trans_type == 'exp':
        mu = torch.exp(theta*d_sim)
        nu = torch.exp(theta*d_obs)
        d = theta * mu
    elif trans_type == 'softplus':
        mu = torch.log(torch.exp(theta*d_sim) + 1)
        nu = torch.log(torch.exp(theta*d_obs) + 1)
        d = theta / torch.exp(-theta*d_sim)
    else:
        mu, nu = d_sim, d_obs
        d = torch.ones(d_sim.shape).to(device)
    mu = mu + c + 1e-18
    nu = nu + c + 1e-18
    return mu, nu, d


def _depth_weighted_loss(m_hat, m):
    rows, cols = m.size()

    # Create a depth weight matrix
    weights = torch.sqrt(torch.arange(rows, dtype=torch.float32) + 8).view(-1, 1)
    weights = weights.cuda()

    # Calculate the difference matrix and perform depth weighting
    diff = m_hat - m
    weighted_diff = weights * diff ** 2

    # Calculate depth weighted error
    weighted_loss = torch.sum(weighted_diff)

    return weighted_loss

def depth_weighted_loss(m_hat, m):
    total_loss = torch.tensor(0.0).cuda()
    for i in range(m_hat.shape[1]):
        total_loss += _depth_weighted_loss(m_hat[:, i, :], m[:, i, :])
    return total_loss

def trace_sum_normalize(x):
    """
    normalization with the summation of each trace
    note that the channel should be 1
    """
    x = x / (x.sum(dim=0,keepdim=True)+1e-18)
    return x

def Wasserstein1(d_sim, d_obs, trans_type='linear', theta=1.1):
    assert d_sim.shape == d_obs.shape
    assert len(d_sim.shape) == 3
    device = d_sim.device
    p = 1
    num_time_steps, num_shots_per_batch, num_receivers_per_shot = d_sim.shape
    d_sim = d_sim.reshape(num_time_steps, num_shots_per_batch * num_receivers_per_shot)
    d_obs = d_obs.reshape(num_time_steps, num_shots_per_batch * num_receivers_per_shot)

    mu, nu, d = transform(d_sim, d_obs, theta)

    assert mu.min() > 0
    assert nu.min() > 0

    mu = trace_sum_normalize(mu)    # normalization operator: N(d)
    nu = trace_sum_normalize(nu)

    F = torch.cumsum(mu, dim=0)
    G = torch.cumsum(nu, dim=0)

    w1loss = (torch.abs(F - G) ** p).sum()  # This function implements equation (5) of the paper
    return w1loss




