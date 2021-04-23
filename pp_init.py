import numpy as np
from itertools import product
import numpy.random as rnd

def pp_params(generation_fn):
    if generation_fn == "stochastic_foraging_session":
        params_dict = {"generation_fn": "stochastic_foraging_session", # stochastic or defined by reward times
                        "generation_params":
                            {
                                "rewsizes":[1,2,4],
                                "N0s":[.125,.25,.5],
                                "n_trials_per_tt": 10,
                                "dt": 0.050,
                                "end_t" : 10
                            }
                        }

    elif generation_fn == "rewtimes_foraging_session":
        rxx = [np.array([0]),np.array([0,1]),np.array([0,2]),np.array([0,1,2])]
        params_dict = {"generation_fn": "rewtimes_foraging_session", # stochastic or defined by reward times
                        "generation_params":
                           {
                               "rewsizes":[2,4],
                               "rewseqs": rxx,
                               "n_trials_per_tt": 15,
                               "dt": 0.050,
                               "end_t" : 10
                           }
                        }

    elif generation_fn == "srw_point_process":
        params_dict = {"generation_fn": "srw_point_process", # stochastic or defined by reward times
                        "generation_params":
                            {
                                "end_t" : 2,
                                "theta0" : 2,
                                "sigma2_eps" : .05,
                                "n_trials": 10,
                                "dt": 0.050
                            }
                        }
    return params_dict

# Point process generation functions (esp) related to foraging session
# Event stream generating functions
def stream_from_events(event_times,end_t,dt = 0.05):
    """
        Convert event times into a binary stream of continuous data
    """
    rew_ix = np.round(event_times / dt).astype(int)
    rew_stream = np.zeros(int(np.round(end_t / dt)))
    rew_stream[rew_ix] = 1
    return rew_stream

def stream_from_sized_events(event_times,event_sizes,end_t,dt = 0.05):
    """
        Approximate rewards of varying sizes by using multiple positive events in a row
    """
    rew_ix = np.round(event_times / dt).astype(int)
    rew_stream = np.zeros(int(np.round(end_t / dt)))
    for ix,dur in zip(rew_ix,event_sizes):
        rew_stream[ix:ix+dur] = 1
    return rew_stream

def srw_point_process(end_t,theta0,sigma2_eps,n_trials = 50,dt = 0.05):
    n_tsteps = int(np.round(end_t / dt))
    theta = np.zeros((n_trials,n_tsteps))
    y = np.zeros((n_trials,n_tsteps))
    for i_trial in range(n_trials):
        theta[i_trial,0] = theta0 + rnd.normal(0,np.sqrt(sigma2_eps))
        for i in range(1,n_tsteps):
            theta[i_trial,i] = theta[i_trial,i-1] + rnd.normal(0,np.sqrt(sigma2_eps))
            lam = np.exp(theta[i_trial,i])
            y[i_trial,i] = int(rnd.rand() < (lam * dt) * np.exp(-lam * dt))
    return y,theta

# Foraging session point process generation functions
def discrete_expo_pdf(N0,n_rew_locs = 20,tau = .125):
    """
        Assign reward delivery probabilities according to scaled Expo decay
    """
    x = np.arange(n_rew_locs)
    cdf = 1 - N0 * np.exp(-tau * x) / tau + N0 / tau
    pdf = cdf[1:] - cdf[:-1]
    return np.insert(pdf,0,1.) # add a deterministic reward at t = 0

def generate_session_y(rewsizes,N0s,n_trials_per_tt,end_t = 20,dt = 0.05):
    """
        Arguments: list of reward sizes and N0 values and # trials per trial type
        Returns: y_tts_trials: dictionary with trial type keys and n_trials_per_tt streams of reward
            - y generated with stream_from_sized_events
    """
    # make PDFs
    pdfs_dict = {}
    for this_n0 in N0s:
        pdfs_dict[this_n0] = discrete_expo_pdf(this_n0,n_rew_locs = end_t)

    # make trial reward streams
    y_tts_trials = {}
    tts = list(product(*[rewsizes,N0s]))
    for (this_rewsize,this_n0) in tts:
        y_tts_trials[(this_rewsize,this_n0)] = np.empty_like([n_trials_per_tt],shape = (n_trials_per_tt,int(round(end_t/dt))))
        for i_trial in range(n_trials_per_tt):
            trial_rewtimes = np.where(rnd.random(end_t) - pdfs_dict[this_n0] < 0)[0]
            y_trial = stream_from_sized_events(trial_rewtimes,np.full(len(trial_rewtimes),this_rewsize),end_t,dt)
            y_tts_trials[(this_rewsize,this_n0)][i_trial,:] = y_trial
    return y_tts_trials

def generate_from_rewtimes_y(rewsizes,rewseqs,n_trials_per_tt,end_t = 20,dt = 0.05):
    """
        Arguments: dict of reward sizes,subplist of  reward time sequences
        Returns: y_tts_trials: list of y reward streams generated from stream_from_sized_events
    """
    y_tts_trials = {}
    tts = list(product(rewsizes,np.arange(len(rewseqs),dtype = int)))
    for (this_rewsize,i_rewseq) in tts:
        y_tts_trials[(this_rewsize,i_rewseq)] = np.tile(stream_from_sized_events(rewseqs[i_rewseq],np.full(len(rewseqs[i_rewseq]),this_rewsize),end_t,dt),(n_trials_per_tt,1))
    return y_tts_trials
