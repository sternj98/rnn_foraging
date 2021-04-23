import numpy as np
from numpy import random as rnd

import torch
import torch.nn as nn

def model_params(offdiag_val = 0.2,diag_val = 0.98,batch_size = 50,nonlinearity = 'relu',readout_nonlinearity = 'linear',n_in = 50,n_out = 1,n_hid =  500):
    """
        Return model kwargs to initialize recurrent model
    """
    model_kwargs = {
                'input_size': n_in,
                'output_size': n_out,
                'batch_size': batch_size,
                'core_kwargs':
                    {
                        'num_layers': 1,
                        'hidden_size': n_hid,
                        'nonlinearity': nonlinearity
                    },
                'param_init_fn': 'diag_init_',
                'param_init_kwargs':
                    {
                        'offdiag_val': offdiag_val / np.sqrt(n_hid),
                        'diag_val': diag_val
                    },
                'readout_nonlinearity': readout_nonlinearity,
                'reservoir_training' : False,
                'reservoir_training_params' :
                    {
                        'weight_lock_step' : 2
                    }
                }
    return model_kwargs

# parameter initialization functions
def diag_init(tensor: torch.Tensor, offdiag_val : np.float64, diag_val: np.float64) -> torch.Tensor:
    """
        Initialize parameters, separating onn and off diag
        In place operation
    """
    shape = tensor.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError('LeInit initializer can only be used for 2D square matrices.')

    off_diag_part = offdiag_val * np.random.randn(shape[0], shape[1])
    tensor = torch.tensor(np.eye(shape[0]) * diag_val + off_diag_part - np.diag(np.diag(off_diag_part)))
    return tensor

def param_init_(recurrent_model):
    param_init_kwargs = recurrent_model.model_kwargs['param_init_kwargs']
#     with torch.no_grad():
    if recurrent_model.model_kwargs['param_init_fn'] == "diag_init_":
        # Init input-hidden according to Xavier Normal, g = 0.95
        nn.init.xavier_normal_(recurrent_model.core.weight_ih_l0,gain = 0.95).double()
        # Init hidden-hidden according to diag-off-diag decomposition method
        recurrent_model.core.weight_hh_l0 = torch.nn.Parameter(diag_init(recurrent_model.core.weight_hh_l0,param_init_kwargs['offdiag_val'],param_init_kwargs['diag_val'])).double()
        # Init hidden-output according to Xavier Normal, g = 0.95
        nn.init.xavier_normal_(recurrent_model.readout_linear.weight,gain = 0.95).double()
        nn.init.xavier_normal_(recurrent_model.readout_sigmoid.weight,gain = 0.95).double()
