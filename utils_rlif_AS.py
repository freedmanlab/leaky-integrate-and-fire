import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

from parameters_rlif_AS import par

num_neurons = par['num_neurons']

def poisson_spikes(n_bins, fr = par['baseline_fr'], dt = par['simulation_dt'], return_n_bin = False, n_bin_offset = 0):
    spike_train = []
    bins_n = []
    for bin in np.arange(n_bins):
        if random.random() < fr * dt:
            fire = 1
            bins_n.append(bin + n_bin_offset)
        else:
            fire = 0
        spike_train.append(fire)
    if return_n_bin == False:
        return spike_train
    else:
        return spike_train, bins_n

def EEG_wave(n_bins, frequency = par['theta_wave_freq'], dt = par['simulation_dt'], n_bin_offset = 0):
    waveform = np.zeros(n_bins)
    bins_n = []
    amplitude = abs(par['V_rest'] - par['V_th']) / 4
    y_offset = amplitude / 2
    for bin in np.arange(n_bins):
        value = amplitude * np.cos(2 * np.pi * frequency * (bin + n_bin_offset) * dt)
        waveform[bin] = np.random.normal(value, par['voltage_stdev']) + y_offset # add in + np.random.normal(0,1)?
    return waveform


def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y

def gudermannian(x):
    y = 2 * np.arctan(np.tanh(x/2))
    return y
