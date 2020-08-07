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

    ##################
    # Leaky intergrate and fire parameters
    'T'                     : 500.0,       # total time to simulate (ms)
    'simulation_dt'         : 0.125,   # Simulation timestep (ms)
    'V_in'                  : .05,      # Neuron input voltage

    'gain'                  : 1.0,      # neuron gain (unitless)
    't_rest'                : 0.05,        # initial refractory time
    't_initial'             : 0.0,        # initial time
    'Rm'                    : 1.0,        # Resistance (kOhm)
    'Cm'                    : 10,       # Capacitance (uF)
    'tau_ref'               : 10.0,        # refractory period (ms)
    'V_th'                  : -.055,     # : 1  #spike threshold
    'V_spike'               : .04,        # spike delta (V)
    'V_rest'                : -.07,        # resting potential (V)
    'type'                  : 'Leaky Integrate and Fire',
    'debug'                 : False,    # Watch neurons get made
    'exc_func'              : default_exc_func, #excitability function
    # Network shape
    'num_layers'            : 1,
    'num_neurons'           : 100,
    'num_inputs'            : 5,
    'input_connect_binary_prob': .3, #governs what fraction of neurons are connected to the inputs
    'input_connect_frac'    : .5, #governs how many inputs neurons that are connected to the inputs are connected to
    'recur_connect_frac'    : .6, #governs connections between neurons
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
    par['tau_m'] = par['Rm'] * par['Cm'] # Time constant

# update_parameters()
update_dependencies()
