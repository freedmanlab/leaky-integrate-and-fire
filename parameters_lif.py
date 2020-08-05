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
    'T'                     : 5.0,       # total time to sumulate (ms)
    'simulation_dt'         : 0.0125,   # Simulation timestep
    'inpt'                  : 1.0,      # Neuron input voltage

    'gain'                  : 1.0,      # neuron gain (unitless)
    't_rest'                : 0.05,        # initial refractory time
    't'                     : 0.0,        # Neuron time step
    'neuron_dt'             : 0.125,    # Neuron dt
    'Rm'                    : 1.0,        # Resistance (kOhm)
    'Cm'                    : 10,       # Capacitance (uF)
    'tau_ref'               : 10.0,        # refractory period (ms)
    'V_th'                  : 0.5,     # : 1  #spike threshold
    'V_spike'               : 1.0,        # spike delta (V)
    'V_rest'                : 0.0,        # resting potential (V)
    'type'                  : 'Leaky Integrate and Fire',
    'debug'                 : False,    # Watch neurons get made
    'exc_func'              : default_exc_func, #excitability function
    # Network shape
    'num_layers'            : 2,
    'num_neurons'           : 100,
    'neuron_connections'    : 3,
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
    par['neuron_input'] = np.full((par['time']),par['inpt'])
    par['tau_m'] = par['Rm'] * par['Cm'] # Time constant

# update_parameters()
update_dependencies()
