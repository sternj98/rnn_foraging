import numpy as np
from numpy import random as rnd
import scipy.stats as stats

lognorm_cdf = lambda x,theta,sigma2: stats.lognorm.cdf(x,s = np.sqrt(sigma2),scale = np.exp(theta))

def ppssm_filtering(y,theta_0,sigma2_0,sigma2_eps,dt = 0.05):
    """
        Calculate estimated rate lambda and its variance
        From a binary sequence y according to PPSSM filtering equations
    """
    seq_len = len(y)
    theta = np.concatenate((np.array([theta_0],dtype = np.float64),np.array(np.zeros(seq_len-1,dtype = np.float64))))
    sigma2 = np.concatenate((np.array([sigma2_0],dtype = np.float64),np.array(np.zeros(seq_len-1,dtype = np.float64))))
    for t in range(1,seq_len):
        sigma2_given_tminus1 = sigma2[t-1] + sigma2_eps
        sigma2[t] = (1 / (dt*np.exp(theta[t-1]) + 1/sigma2_given_tminus1))
        theta[t] = theta[t-1] + sigma2[t] * (y[t] - dt*np.exp(theta[t-1]))
    return theta,sigma2

# Analyze PPSSM results as probabilities
def lamda_posterior_hmap(theta_trial,sigma2_trial,pdf_xvals):
    """
        Given theta and sigma2 PPSSM estimates from a trial, return the lognormal posterior per timepoint
    """
    posterior = np.zeros((len(pdf_xvals),len(sigma2_trial)))
    for t in range(len(sigma2_trial)):
        posterior[:,t] = stats.lognorm.pdf(pdf_xvals,s = np.sqrt(sigma2_trial[t]),scale = np.exp(theta_trial[t]))
    return posterior

def p_lessthan_x(ppssm_tts_trials,x):
    """
        Given PPSSM results for a session, map trials into P(lambda < x) using the lognormal CDF
    """
    p_lessthan_tts_trials = {}
    for tt in ppssm_tts_trials.keys():
        p_lessthan_tts_trials[tt] = np.zeros((len(ppssm_tts_trials[tt]),len(ppssm_tts_trials[tt][0][0])))
        for i_trial in range(len(ppssm_tts_trials[tt])):
            p_lessthan_tts_trials[tt][i_trial,:] = lognorm_cdf(x,ppssm_tts_trials[tt][i_trial][0],ppssm_tts_trials[tt][i_trial][1])
    return p_lessthan_tts_trials

def stochastic_prt_gen(p_lessthan_tts_trials,prt_lock = None,dt = 0.05):
    """
        Given p_lessthan_tts_trials, return stochastically generated patch residence times
        Using p_lessthan_tts_trials as p(leave)

        Add PRT lock to ensure that agent stays until x time
    """
    prts = {}
    for i_tt,tt in enumerate(p_lessthan_tts_trials.keys()):
        t_len = len(p_lessthan_tts_trials[tt][0])
        prts[tt] = np.zeros(len(p_lessthan_tts_trials[tt]))
        for i_trial in range(len(prts[tt])):
            leave_ix = np.where(rnd.random(t_len) - dt * p_lessthan_tts_trials[tt][i_trial,:] < 0)[0]
            if prt_lock == None:
                if len(leave_ix) > 0:
                    prts[tt][i_trial] = dt * leave_ix[0]
                else:
                    prts[tt][i_trial] = dt * t_len
            else:
                if len(leave_ix) > 0:
                    if len(leave_ix[leave_ix*dt > prt_lock[i_tt]]): # lock prt to min value per tt
                        prts[tt][i_trial] = dt * leave_ix[leave_ix*dt > prt_lock[i_tt]][0]
                    else:
                        prts[tt][i_trial] = dt * t_len
                else:
                    prts[tt][i_trial] = dt * t_len
    return prts
