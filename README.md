# FWI-WF

This repo contains a PyTorch implementation with Deepwave for the paper [FWI-WF: Robust full waveform inversion using hybrid Wasserstein-1 and Fourier metrics](), which is submitted to the Journal of Geophysics. 

## Prerequisites
```
python 3.9.4 
pytorch 1.7.1
scipy 1.8.0
numpy 
matplotlib 3.5.1
scikit-image 0.19.2
math
jupyter
pytest 6.2.4
IPython
deepwave 0.0.9
```
#### NOTE 
## Install deepwave 0.0.9
1. cd  ./FWI-WF/FWI-WF/
When installing deepwave, you need deepwave/, README.md, setup.py, setup.cfg
2. python setup.py install

Successful installation will result in the 'build', 'deepwave.egg-info', and 'dist' folders.
We install the deepwave which provides finite difference scheme with 8th-order accuracy in the space domain. 


## An example
Case 1: Inversion results of the Marmousi model using noise-free data
```
mar_smal.ipynb
```

## Cross-linking (code -> paper)
loss_w = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true)  ->  Eq. (7)
loss_fft = FFT_loss(u_obs, u_pred)  ->  Eq. (13)
weight = 1.0 / (1.0 + np.exp(-(epoch - num_epochs // 2) / 10.0))  -> Eq. (15)
loss = weight * loss_fft + (1.0 - weight) * loss_w   ->   Eq. (14)


