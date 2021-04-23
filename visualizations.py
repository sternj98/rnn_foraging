import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import zscore

def peaksort_vis_trial(core_activity : torch.Tensor,normalization = "max",vis_end_ix = None,sort_end_ix = None,ax = None):
    """
        Peaksort visualization of network activity during task
    """
    if vis_end_ix == None:
        core_activity = core_activity.detach().numpy().T
    else:
        core_activity = core_activity.detach().numpy().T[:,:vis_end_ix]
    core_activity = core_activity[np.any(core_activity!= 0,1),:]
    if sort_end_ix:
        ix = np.argmax(core_activity[:,:sort_end_ix],1)
    else:
        ix = np.argmax(core_activity,1)
    peaksort = np.argsort(ix)
    if normalization == "max":
        if ax:
            ax.imshow(np.flipud(core_activity[peaksort,:] / np.max(core_activity,1)[peaksort,np.newaxis]),cmap = 'jet')
        else:
            plt.imshow(np.flipud(core_activity[peaksort,:] / np.max(core_activity,1)[peaksort,np.newaxis]),cmap = 'jet')
    elif normalization == "zscore":
        if ax:
            ax.imshow(zscore(np.flipud(core_activity[peaksort,:]),axis = 1),cmap = 'jet')
        else:
            plt.imshow(zscore(np.flipud(core_activity[peaksort,:]),axis = 1),cmap = 'jet')
    return peaksort

def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
