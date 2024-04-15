"""
Common functions for plotting figures
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotcomparison(gt,pre,ite,SaveFigPath):
    """
    plot current inverted model and ground truth
    """
   
    dim = gt.shape
    gt = gt.reshape(dim[0],dim[1])
    pre = pre.reshape(dim[0],dim[1])
        
    vmin, vmax = np.percentile(gt, [2,98])
    fig1, ax1 = plt.subplots(2)
    fig1.set_figheight(6)
    fig1.set_figwidth(12)
    
    plt.subplot(1,2,1) 
    plt.imshow(pre,vmin=vmin, vmax=vmax, cmap='jet')
    plt.colorbar()
    plt.title('inverted model')
    
    plt.subplot(1,2,2) 
    plt.imshow(gt, vmin=vmin, vmax=vmax, cmap='jet')
    plt.colorbar()
    plt.title('true model')
    
    plt.savefig(SaveFigPath+'invert_ite{}.png'.format(ite))

    plt.close()
    
    
def plotinitsource(init,gt,SaveFigPath):
    """
        plot initial and true source amplitudes
    """
    t = 500
    figsize = (12, 6)
    plt.figure(figsize=figsize)
    plt.plot(init.reshape(-1).ravel()[:t], label='Initial')
    plt.plot(gt.reshape(-1).ravel()[:t], label='True')
    plt.legend()
    
    plt.title('source amplitude')
    
    plt.savefig(SaveFigPath+'source.png')

    plt.close()


def PlotSNR(SNR, SaveFigPath):
    """
       Plot SNR between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(SNR[1:],color='purple',lw=1,ls='-',marker='v',markersize=2,label='SNR') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('SNR')
    plt.savefig(str(SaveFigPath)+'SNR.png')
    plt.close()

    
def PlotRSNR(RSNR, SaveFigPath):
    """
       Plot RSNR between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(RSNR[1:],color='green',lw=1,ls='-',marker='v',markersize=2,label='RSNR') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('RSNR')
    plt.savefig(str(SaveFigPath)+'RSNR.png')
    plt.close()

    
def PlotSSIM(SSIM, SaveFigPath):
    """
       Plot SSIM between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(SSIM[1:],color='green',lw=1,ls='-',marker='v',markersize=2,label='SSIM') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('SSIM')
    plt.savefig(str(SaveFigPath)+'SSIM.png')
    plt.close()
    
    
def PlotERROR(ERROR,SaveFigPath):
    """
       Plot ERROR between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(ERROR[1:],color='green',lw=1,ls='-',marker='v',markersize=2,label='ERROR') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('ERROR')
    plt.savefig(str(SaveFigPath)+'ERROR.png')
    plt.close()
    
