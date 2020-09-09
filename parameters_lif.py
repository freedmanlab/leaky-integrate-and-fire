import numpy as np
import tensorflow as tf
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

    ##################
    # Leaky intergrate and fire parameters
    'T'                     : 0.5,       # total time to sumulate (s)
    'simulation_dt'         : 0.001,   # Simulation timestep (1 ms)

    'gain'                  : 1.0,      # neuron gain (unitless)
    't_rest_absolute'       : 0.002,        # absolute refractory time; biological: 1-2ms
    't_rest_relative'       : 0.004,        # relative refractory time (hyperpolarization); biological: 3-4ms
    'neuron_dt'             : 0.001,    # Neuron dt
    'Rm'                    : 3e8,        # Resistance (Ohm); biological: 300-500 MOhm
    'Cm'                    : 0.9e-6,       # Capacitance (F); biological: 0.9 uF
    'tau_ref'               : 10.0,        # refractory period (ms)
    'V_th'                  : -55,     #spike threshold (mV); biological: -55 mV
    'V_spike'               : 40,        # spike delta (mV); biological: 40 mV
    'V_rest'                : -70,        # resting potential (mV); biological: -70 mV
    'V_hyperpolar'          : -90,      # hyperpolarization potential (mV); biological: -90 mV
    'type'                  : 'Leaky Integrate and Fire',
    'debug'                 : False,    # Watch neurons get made
    'exc_func'              : default_exc_func, #excitability function

    'baseline_fr'  : 6, # Baseline neuron firing rate (Hz); biological: 6-100 Hz

    # Synaptic plasticity
    'synaptic_plasticity'   : True, # No option for False currently
    'n_std_devs'            : 5, # number of standard deviations from middle to ends of neuronal array for synaptic plasticity starting values
    'syn_plas_constant'     : 1e-5, # weight multiplier of synaptic plasticity values

    # Network shape
    'n_hidden'              : 5, # does NOT include input or output layer; this is number of hidden layers
    'num_neurons'           : 100,
    'num_input_neurons'     : 8, # number of input neurons. These somehow encode task information, firing rate?
    'num_output_neurons'    : 2, # number of output neurons. These somehow encode task information, firing rate?
    'neuron_connections'    : 3, # only used if synaptic plasticity is False
    'excitability_ratio'    : 0.8, # percentage of excitabliity neurons, inhibitory percentage = 1-excitability_ratio

    # Task information
    'task_info'             : 'DMS', # Options: delayed match to sample (DMS), baseline firing (baseline_firing), None
    'preferred_angle_fr'    : 75, # Firing rate at the preferred angle; biological: 75 Hz

    # Trial params
    'p_match_sample_eq'     : 0.25, # Probability that match and sample direction are equal
    'lr_w'                  : 0.01, # Learning rate for weights
    'lr_th'                 : 0.01, # Learning rate for thresholds
}

def update_parameters(updates):
    """ Takes a list of strings and values for updating parameters in the parameter dictionary
        Example: updates : [(key, val), (key, val)] """

    for key, val in updates.items():
        par[key] = val

    update_dependencies()

def update_dependencies():
    """ Updates all parameter dependencies """
    par['time'] = int(par['T'] / par['simulation_dt'])
    par['inpt'] = par['V_spike'] * 1.5
    par['neuron_input'] = np.full((par['time']), par['inpt'])
    par['tau_m'] = par['Rm'] * par['Cm'] # Time constant

    # Start with all neurons projecting
    if par['synaptic_plasticity']:
        par['neuron_connections'] = par['num_neurons']

    if par['task_info'] == 'DMS': # update all values for DMS
        par['T'] = 3.0
        par['num_input_neurons'] = 8
        par['num_output_neurons'] = 2

# update_parameters()
update_dependencies()
