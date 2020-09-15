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

def spikes_to_current(time, tau):
    return .0101*(np.exp(-(time/(4*tau)))-np.exp(-(time/tau)))

par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'save_fn'               : 'model_results.pkl',

    ##################
    # Leaky intergrate and fire parameters
    'T'                     : 300.0,       # total time to simulate (ms)
    'simulation_dt'         : 0.125,   # Simulation timestep (ms)
    'V_in'                  : .05,      # Neuron input voltage
    'input_duration'        : 8.0,     # duration of the input current (ms)

    'gain'                  : 1.0,      # neuron gain (unitless)
    't_rest'                : 0.05,        # initial refractory time
    'Rm'                    : 1.0,        # Resistance (kOhm)
    'Cm'                    : 10,       # Capacitance (uF)
    'tau_ref'               : 5.0,        # refractory period (ms)
    'tau_abs_ref'           : 1.0,        # absolute refractory period (ms), lower bound for exc_refrac
    'V_th'                  : -.055,     # : 1  #spike threshold
    'V_spike'               : .04,        # spike delta (V)
    'V_rest'                : -.07,        # resting potential (V)
    'type'                  : 'Leaky Integrate and Fire',
    'debug'                 : False,    # Watch neurons get made
    'exc_func'              : default_exc_func, #excitability function
    'input_stdev'           : .001,     #standard deviation of the input Gaussian noise
    'voltage_stdev'         : .0008,     #standard deviation of the neuron voltage update, also Gaussian
    'spikes_to_current_func': spikes_to_current, #function for converting spikes to current
    'voltage_decay_const'   : .5,        #decay constant for the conversion from spikes to current (ms)
    'decay_thresh'          : .0005,    #threshold for zeroing the current from a spike
    # Network shape
    'num_layers'            : 1,
    'num_neurons'           : 100,
    'num_inputs'            : 5,
    'exc_prop'              : .8,     #proportion of neurons that are excitatory
    'input_connect_binary_prob': .3, #governs what fraction of neurons are connected to the inputs
    'input_connect_frac'    : .3, #governs how many inputs neurons that are connected to the inputs are connected to
    'recur_connect_frac'    : .3, #governs connections between neurons
}

def update_parameters(updates):
    """ Takes a list of strings and values for updating parameters in the parameter dictionary
        Example: updates : [(key, val), (key, val)] """

    for key, val in updates.items():
        par[key] = val

    update_dependencies()

def update_dependencies():
    """ Updates all parameter dependencies """
    par['timesteps'] = int(par['T'] / par['simulation_dt'])
    par['input_timesteps'] = int(par['input_duration']/par['simulation_dt'])
    par['tau_m'] = par['Rm'] * par['Cm'] # Time constant
    par['num_current_timesteps'] = compute_current_timesteps()
    par['exc_num'] = int(par['exc_prop']*par['num_neurons'])

def compute_current_timesteps():
    times = par['simulation_dt']*np.arange(1000)
    currents = spikes_to_current(times, par['voltage_decay_const'])
    return 1000-np.argmax(currents[::-1]>par['decay_thresh'])
# update_parameters()
update_dependencies()
