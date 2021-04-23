import numpy as np
from numpy import random as rnd
import scipy.io as sio
from itertools import product
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from scipy.stats import zscore

from model_init import diag_init,param_init_

class RecurrentModel(nn.Module):
    """
        Generalized RNN model from Rylan Schaeffer
    """
    def __init__(self,model_kwargs):
        super(RecurrentModel, self).__init__()
        self.model_kwargs = model_kwargs
        self.input_size = model_kwargs['input_size']
        self.output_size = model_kwargs['output_size']
        self.batch_size = model_kwargs['batch_size']

        # create and save core i.e. the recurrent operation
        self.core = self._create_core(model_kwargs=model_kwargs)

        self.readout_linear = nn.Linear( # linear output layer
            in_features=model_kwargs['core_kwargs']['hidden_size'],
            out_features=self.output_size,
            bias=True)
            
        self.readout_sigmoid = nn.Linear(
            in_features=model_kwargs['core_kwargs']['hidden_size'],
            out_features=self.output_size,
            bias=True)

        param_init_(self)
        # only train readout
        # if self.model_kwargs['reservoir_training'] == True:
        #     for i_param in self.core.parameters():
        #         i_param.requires_grad = False

        self.reset_core_hidden()

        # converts all weights into doubles i.e. float64
        # this prevents PyTorch from breaking when multiplying float32 * float64
        self.double()

    def _create_core(self, model_kwargs):
        core = nn.RNN(
            input_size=self.input_size,
            batch_first=True,
            **model_kwargs['core_kwargs'])
        return core

    def _create_readout_nonlinearity(self,model_kwargs):
        """
            Currently deprecated
        """
        if model_kwargs['readout_nonlinearity'] == "sigmoid":
            return torch.sigmoid
        elif model_kwargs['readout_nonlinearity'] == "linear":
            return torch.nn.Identity(self.model_kwargs['core_kwargs']['hidden_size'])
        elif model_kwargs['readout_nonlinearity'] == "elu":
            return torch.nn.ELU(self.model_kwargs['core_kwargs']['hidden_size'])

    def reset_core_hidden(self):
        self.core_hidden = torch.zeros(self.output_size,self.batch_size,self.model_kwargs['core_kwargs']['hidden_size'],dtype = torch.double)

    def forward(self,x):
        """
        Performs a forward pass through model.

        :param model_input: Tensor with shape (batch size, num step, stimulus dimension)

        :return forward_output: dictionary containing 4 keys:
            core_output: Tensor of shape (batch size, num steps, core dimension)
            readout_output: Tensor of shape (batch size, num steps, core dimension)
            readout: Tensor of shape (batch size, num steps, output dimension)
        """
        # reset hidden
        self.reset_core_hidden()

        # run trials and get core output
        core_output, new_core_hidden = self.core(x,self.core_hidden)

        readout_output = self.readout_linear(core_output)
        sigmoid_readout_output = torch.sigmoid(self.readout_sigmoid(core_output))

        forward_output = dict(
            core_output = core_output,
            readout_output = readout_output,
            sigmoid_readout_output = sigmoid_readout_output)

        return forward_output
