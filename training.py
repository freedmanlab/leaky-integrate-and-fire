import numpy as np
import random
# from model_lif import exc_func

from parameters_lif import par

num_neurons = par['num_neurons']

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

def loss_function(values, targets, loss_function = 'quadratic'):
    if par['task_info'] == 'DMS':
        loss = np.linalg.norm(values - targets)
    return loss

def backprop(neurons, synaptic_weights, lr_w = par['lr_w']):
    pass

    for layer in reversed(np.arange(1, par['n_hidden'] + 1)):
        for output_neuron in np.arange(num_neurons):
            for input_neuron in np.arange(num_neurons):
                N_l = num_neurons # number of synapses of each neuron
                m_l = layer
                delta_i_l = g_i_l / g_hat_i_l * np.sqrt(M_l_plus_1 / m_l_plus_1)
                synaptic_weights[layer][output_neuron][input_neuron] = -1 * lr_w * np.sqrt(N_l / m_l ) * delta_i_l * x_hat_j_l
