import numpy as np
import tensorflow as tf
import os

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'save_fn'               : 'model_results.pkl',

    ##################
    # Leaky intergrate and fire parameters
    'T'                     : 50,       # total time to sumulate (msec)
    'simulation_dt'         : 0.0125,   # Simulation timestep
    'inpt'                  : 1.0,      # Neuron input voltage

    't_rest'                : 0,        # initial refractory time

    't'                     : 0,        # Neuron time step
    'neuron_dt'             : 0.125,    # Neuron dt
    'Rm'                    : 1,        # Resistance (kOhm)
    'Cm'                    : 10,       # Capacitance (uF)
    'tau_ref'               : 4,        # refractory period (ms)
    'Vth'                   : 0.75,     # : 1  #spike threshold
    'V_spike'               : 1,        # spike delta (V)
    'type'                  : 'Leaky Integrate and Fire',
    'debug'                 : False,    # Watch neurons get made

    # Network shape
    'num_layers'            : 4,
    'num_neurons'           : 1024,
}

def update_parameters(updates):
    """ Takes a list of strings and values for updating parameters in the parameter dictionary
        Example: updates : [(key, val), (key, val)] """

    for key, val in updates.items():
        par[key] : val

    update_dependencies()

def update_dependencies():
    """ Updates all parameter dependencies """
    par['time']           = int(par['T'] / par['simulation_dt'])
    par['neuron_input']   = np.full((par['time']),par['inpt'])
    par['tau_m']          = par['Rm'] * par['Cm'] # Time constant

# update_parameters()
update_dependencies()
