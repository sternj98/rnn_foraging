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

from pp_init import generate_session_y,generate_from_rewtimes_y,srw_point_process
from ppssm_utils import ppssm_filtering

class Task(object):
    """
        Iterable task object
    """
    def __init__(self, max_iter=None, batch_size=1):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1) , self.sample()
        else:
            raise StopIteration()

    def sample(self):
        raise NotImplementedError()

class KalmanFilteringTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigtc_sq=4.0, signu_sq=1.0, gamma=0.1, tr_cond='all_gains'):
        super(KalmanFilteringTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size
        self.n_in       = n_in
        self.n_out      = n_out
        self.stim_dur   = stim_dur
        self.sigtc_sq   = sigtc_sq
        self.signu_sq   = signu_sq
        self.gamma      = gamma # controls how much the hidden state is going to change per timestep
        self.tr_cond    = tr_cond # how reliable is sensory information (not of huge importance to us either)
        self.phi        = np.linspace(-9.0, 9.0, self.n_in) # Phi is the input tuning function

    def sample(self):

        NU         = np.sqrt(self.signu_sq) * np.random.randn(self.stim_dur, self.batch_size)
        R          = np.zeros((self.n_in, self.stim_dur, self.batch_size))
        S          = np.zeros((1, self.stim_dur, self.batch_size))
        M          = np.zeros((1, self.stim_dur, self.batch_size))
        SIG_SQ     = np.zeros((1, self.stim_dur, self.batch_size))
        M_IN       = np.zeros((1, self.stim_dur, self.batch_size))
        SIG_SQ_IN  = np.zeros((1, self.stim_dur, self.batch_size))

        A_in       = np.ones((1, self.n_in))
        B_in       = self.phi

        if self.tr_cond == 'all_gains':
            G         = (3.0 - 0.3) * np.random.rand(self.stim_dur, self.batch_size) + 0.3
        elif self.tr_cond == 'high_gain':
            G = np.full((self.stim_dur, self.batch_size),3)
        else:
            G         = np.random.choice([0.3, 3.0],(self.stim_dur, self.batch_size))

        for ii in range(self.batch_size):
            S[0,0,ii]         = np.sqrt(self.signu_sq) * np.random.randn()
            R[:,0,ii]         = G[0,ii] * np.exp(- ((S[0,0,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
            R[:,0,ii]         = np.random.poisson(R[:,0,ii])
            M[0,0,ii]         = np.dot(B_in, R[:,0,ii]) / (np.dot(A_in, R[:,0,ii]) + (self.sigtc_sq/self.signu_sq))
            SIG_SQ[0,0,ii]    = 1.0 / ( np.dot(A_in, R[:,0,ii]) / self.sigtc_sq + (1.0 / self.signu_sq))
            M_IN[0,0,ii]      = M[0,0,ii]
            SIG_SQ_IN[0,0,ii] = SIG_SQ[0,0,ii]

            for tt in range(1,self.stim_dur):
                # Draw the new mean of the generating process s_t
                S[0,tt,ii]         = (1.0 - self.gamma) * S[0,tt-1,ii] + NU[tt,ii]
                # Draw new stimulus input through poisson draws w/ rate drawn according to tuning function
                R[:,tt,ii]         = G[tt,ii] * np.exp(- ((S[0,tt,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
                R[:,tt,ii]         = np.random.poisson(R[:,tt,ii])

                natparam_1_in      = np.dot(B_in, R[:,tt,ii]) / self.sigtc_sq
                natparam_2_in      = np.dot(A_in, R[:,tt,ii]) / self.sigtc_sq

                M_IN[0,tt,ii]      = natparam_1_in / natparam_2_in
                SIG_SQ_IN[0,tt,ii] = 1.0 / natparam_2_in

                K                  = self.signu_sq + (1.0-self.gamma)**2 * SIG_SQ[0,tt-1,ii]

                M[0,tt,ii]         = ( np.dot(B_in, R[:,tt,ii]) * K + (1.0-self.gamma) * M[0,tt-1,ii] * self.sigtc_sq) / ( np.dot(A_in, R[:,tt,ii]) * K + self.sigtc_sq)
                SIG_SQ[0,tt,ii]    = (self.sigtc_sq * K) / (np.dot(A_in, R[:,tt,ii]) * K + self.sigtc_sq)

        example_input         = np.swapaxes(R,0,2)
        example_output        = np.swapaxes(S,0,2)
        opt_s                 = np.swapaxes(M,0,2)

        return torch.from_numpy(example_input).double() , torch.from_numpy(example_output).double(),opt_s

class PPSSM_FilteringTask(Task):
    """
        Filter point process to return estimated lambda

        Use foraging session function to create the generative point process (S)
    """
    def __init__(self,ppssm_params,max_iter=100, n_in=50,sigtc_sq=.02,theta0 = 4,sigma2_0 = 0.1,sigma2_eps = 0.04,tr_cond='high_gain'):
        super(PPSSM_FilteringTask, self).__init__(ppssm_params)
        self.n_in       = n_in
        self.ppssm_params = ppssm_params
        self.max_iter = max_iter
        self.y_generator, self.stim_dur, self.batch_size = self._create_PPSSM_generating_fn()
        self.theta0 = theta0
        self.sigma2_0 = sigma2_0
        self.sigma2_eps = sigma2_eps

        self.tr_cond    = tr_cond # how reliable is sensory information
        # Phi is the input tuning function
        self.phi        = np.linspace(0, 1, self.n_in) # For some reward cells and some no-reward cells
#         self.phi        = np.ones(self.n_in) # Input only to reward
        self.dt = self.ppssm_params['generation_params']['dt']
        self.sigtc_sq = sigtc_sq

    def _create_PPSSM_generating_fn(self):
        gen_fn_params = self.ppssm_params["generation_params"]
        if self.ppssm_params["generation_fn"] == "stochastic_foraging_session":
            y_generator = generate_session_y
            n_tts = len(list(product(*[gen_fn_params['rewsizes'],gen_fn_params['N0s']])))
            stim_dur = int(np.round(gen_fn_params['end_t'] / gen_fn_params['dt']))
            batch_size = int(gen_fn_params['n_trials_per_tt'] * n_tts)
        elif self.ppssm_params["generation_fn"] == "rewtimes_foraging_session":
            y_generator = generate_from_rewtimes_y
            n_tts = len(list(product(*[gen_fn_params['rewsizes'],gen_fn_params['rewseqs']])))
            stim_dur = int(np.round(gen_fn_params['end_t'] / gen_fn_params['dt']))
            batch_size = int(gen_fn_params['n_trials_per_tt'] * n_tts)
        elif self.ppssm_params["generation_fn"] == "srw_point_process":
            y_generator = srw_point_process
            stim_dur = int(np.round(gen_fn_params['end_t'] / gen_fn_params['dt']))
            batch_size = gen_fn_params['n_trials']
        return y_generator,stim_dur,batch_size

    def sample(self):
        # Set trial stimulus gains
        if self.tr_cond == 'all_gains':
            G         = (3.0 - 0.3) * np.random.rand(self.stim_dur, self.batch_size) + 0.3
        elif self.tr_cond == 'high_gain':
            G         = np.full((self.stim_dur, self.batch_size),3)
        else:
            G         = np.random.choice([0.3, 3.0],(self.stim_dur, self.batch_size))

        # make session and flatten it to get training batch
        y_dict     = self.y_generator(**self.ppssm_params['generation_params'])
        if type(y_dict) == dict: # for the session generation functions
            S          = np.concatenate([y_dict[i] for i in y_dict.keys()]).T[np.newaxis,:,:]
        else:
            S = y_dict[0].T[np.newaxis,:,:]
        self.sigma2_eps  = .04 # np.var(S)

        # initialize input neuron response tensor
        R          = np.zeros((self.n_in, self.stim_dur, self.batch_size))

        # Initialize parameters for optimal filtering from sensory input
        M          = np.zeros((1, self.stim_dur, self.batch_size))
        SIG_SQ     = np.zeros((1, self.stim_dur, self.batch_size))

        ones    = np.ones((1, self.n_in))

        for ii in range(self.batch_size): # iterate over trials
            # initialize filtering stuff for this trial
            R[:,0,ii]         = G[0,ii] * np.exp(- ((S[0,0,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
            R[:,0,ii]         = np.random.poisson(R[:,0,ii])
            M[0,0,ii]         = 2 # np.dot(self.phi, R[:,0,ii]) / np.dot(ones,self.phi)
            SIG_SQ[0,0,ii]    = .5 # initialize to relatively high volatility self.sigma2_eps

            theta,sigma2 = ppssm_filtering(S[0,:,ii],self.theta0,self.sigma2_0,self.sigma2_eps)
            M[0,:,ii] = theta
            SIG_SQ[0,:,ii] = sigma2

            for tt in range(1,self.stim_dur): # iterate over timesteps
                # Draw new stimulus input through poisson draws w/ rate drawn according to tuning function
                R[:,tt,ii]         = G[tt,ii] * np.exp(- ((S[0,tt,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
                R[:,tt,ii]         = rnd.poisson(R[:,tt,ii])

                # y_obs = np.dot(self.phi, R[:,tt,ii]) / (np.dot(ones,self.phi) * G[tt,ii])
                # y_obs = S[0,tt,ii]
                # sigma2_given_tminus1 = SIG_SQ[0,tt-1,ii] + self.sigma2_eps
                # SIG_SQ[0,tt,ii] = 1 / (self.dt * np.exp(M[0,tt-1,ii]) + 1 / sigma2_given_tminus1)
                # M[0,tt,ii] = M[0,tt-1,ii] + SIG_SQ[0,tt,ii] * (y_obs - self.dt * np.exp(M[0,tt-1,ii]))

        example_input         = np.swapaxes(R,0,2) # this will be the poisson drawn input
        example_output        = np.swapaxes(S,0,2) # this will be the stream of reward events (ground truth)
        opt_s                 = np.swapaxes(M,0,2) # this will be output of PPSSM call
        opt_s_sigma2          = np.swapaxes(SIG_SQ,0,2)

        return torch.from_numpy(example_input).double() , torch.from_numpy(example_output).double(),opt_s,opt_s_sigma2
