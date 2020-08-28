import numpy as np
import random
# from model_lif import exc_func

from parameters_lif import par

def poisson_spikes(n_bins, fr = par['baseline_fr'], dt = par['simulation_dt'], return_n_bin = False):
    spike_train = []
    bins_n = []
    for bin in np.arange(n_bins):
        if random.random() < fr * dt:
            fire = 1
            bins_n.append(bin)
        else:
            fire = 0
        spike_train.append(fire)
    if return_n_bin == False:
        return spike_train
    else:
        return spike_train, bins_n
