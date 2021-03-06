import numpy as np
#import tensorflow as tf
import os

print("--> Loading parameters...")

"""
Independent parameters
"""

#define the default excitability function
def default_exc_func(V_rest, V_th, tau_ref, gain, V_m, spikes, I, exc):
    return V_rest, V_th, tau_ref, gain

par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'save_fn'               : 'model_results.pkl',
    'set_seed'              : 42, # Random seed set for reproduction. Either numerical value or False

    ##################
    # Leaky intergrate and fire parameters
    'T'                     : 2.0,       # total time to simulate (s)
    'simulation_dt'         : 0.001,   # Simulation timestep (1 ms)
    'V_in'                  : -50,      # Neuron input voltage
    'input_duration'        : 12.0,     # duration of the input current (s) # DEPRECATED

    'gain'                  : 1.0,      # neuron gain (unitless)
    'Rm'                    : 10,        # Resistance (MOhm); biological: ~10 MOhm
    'Cm'                    : .001,       # Capacitance (uF); biological: 0.9 uF
    'tau_exc'               : .100,      # time constant for decay of exc properties (s), >> tau_m
    'tau_ref'               : 0.005,        # refractory period (s); biological: 3-5 ms
    'tau_abs_ref'           : 0.001,        # absolute refractory period (s), lower bound for exc_refrac
    'V_th'                  : -55,     # spike threshold (mV); biological: -55 mV
    'V_spike'               : 40,        # spike delta (mV); biological: 40 mV
    'V_rest'                : -70,        # resting potential (mV); biological: -70 mV
    'V_hyperpolar'          : -90,        # hyperpolarization potential (mV); biological: -90 mV
    'type'                  : 'Leaky Integrate and Fire',
    'debug'                 : False,    # Watch neurons get made
    'exc_func'              : default_exc_func, # excitability function
    'input_stdev'           : 3,     # standard deviation of the input Gaussian noise; biological: idk, # TODO: Find out. Should still be same as voltage stdev?
    'voltage_stdev'         : 3,     # standard deviation of the neuron voltage update, also Gaussian; biological: 3-3.25 (Baroni et al., 2014)
    # 'spikes_to_spikepulse_func': spikes_to_spikepulse, # function for converting spikes to a pulse
    'voltage_decay_const'   : .005,        # decay constant for the conversion from spikes to a pulse (ms)
    'decay_thresh'          : .0005,    # threshold for zeroing the current from a spike

    # Network shape
    'num_layers'            : 1, # just one recurrent layer
    'num_neurons'           : 100,
    'num_inputs'            : 5,
    'exc_prop'              : .7,     # proportion of neurons that are excitatory
    'input_connect_binary_prob': .3, # governs what fraction of neurons are connected to the inputs
    'input_connect_frac'    : .3, # governs how many inputs neurons that are connected to the inputs are connected to
    'recur_connect_frac'    : .3, # governs connections between neurons
    'inhib_weight'          : -3, # value for weight of inhibitory neurons

    # Memory inputs
    'baseline_fr'           : 3, # Baseline neuron firing rate (Hz); biological: 3-10 Hz (theta waves)
    'theta_wave_freq'       : 6, # Frequency of the theta waves (Hz); biological: 6-10
    'input_1_freq'          : 10, # Frequency of the first input (Hz) # place cells
    'input_2_freq'          : 100, # Frequency of the second input (Hz) # grid cells

    # Excitability parameters
    'exc_rest_max'          : -60, # maximum resting potential (mV); biological: idk, # TODO: Find out
    'exc_thresh_min'        : -60, # minumum threshold potential (mV); biological: idk, # TODO: Find out
    'num_relevant_timebins' : 200, # number of time bins used to calculate excitability

    'spike_offset'          : 10, # By how much to shift excitability functions to the right; 10 is a good value, shows spike trains

    'timedep_scale'         : "linear", # scale of time dependency for excitability. Options: linear, geometric (aka logarithmic)
    'timedep_min_weight'    : 0, # the minimum weight of spikes, spikes farther away weighted less
    'timedep_max_weight'    : 2, # the maximum weight of spikes, more recent spikes weighted up to this value. 2 is a good value, do not change

    'tau_exc_long'          : 0.5, # longer refractory time for excitability

    'neurotransmitters'     : ['AMPA', 'NMDA', 'GABA'],

    'AMPA_spikeoffset_short': 2, # spike offset for increasing probability of open AMPA receptors
    'AMPA_spikeoffset_long' : 7, # spike offset for increasing density of AMPA receptors at membrane

    'NMDA_offset_long'      : 12, # spike offset for increasing NMDA LTP

    'GABA_spikeoffset_short': 2, # spike offset for increasing probability of open GABA receptors
    'GABA_spikeoffset_long' : 7, # spike offset for increasing density of GABA receptors at membrane
}

def update_parameters(updates):
    """ Takes a list of strings and values for updating parameters in the parameter dictionary
        Example: updates : [(key, val), (key, val)] """

    for key, val in updates.items():
        par[key] = val

    update_dependencies()

def spikes_to_spikepulse(time, tau):
    return 0.0101*(np.exp(-(time/(4*tau)))-np.exp(-(time/tau))) # * 1000 # convert to mV

def update_dependencies():
    """ Updates all parameter dependencies """
    par['timesteps'] = int(par['T'] / par['simulation_dt'])
    par['input_timesteps'] = int(par['input_duration'] / par['simulation_dt'])
    par['tau_m'] = par['Rm'] * par['Cm'] # Time constant
    par['num_spikepulse_timesteps'] = compute_spikepulse_timesteps()
    par['exc_num'] = int(par['num_neurons'] * par['exc_prop'])
    par['neuron_receptor_weights'] = [(par['exc_prop'])/2, (par['exc_prop'])/2, 1-par['exc_prop']]

def compute_spikepulse_timesteps(): # Compute how long each spike pulse lasts. Assume it's less than 1000
    times = par['simulation_dt']*np.arange(1000)
    currents = spikes_to_spikepulse(times, par['voltage_decay_const'])
    return 1000-np.argmax(currents[::-1]>par['decay_thresh'])
# update_parameters()
update_dependencies()
print("--> Parameters loaded successfully")
