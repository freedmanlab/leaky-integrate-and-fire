import numpy as np
import math

from parameters_rlif_AS import par
from utils_rlif_AS import sigmoid, gudermannian

dt          = par['simulation_dt'] # Simulation timestep
tau_exc     = par['tau_exc']

# define resting excitability function - params are V_rest, V_m, spikes, I, exc

# make a spike rate increase function and a spike rate decrease function I think
def exc_static_up_func(V_rest, V_th, tau_ref, gain, V_m, spikes, input, exc):
    # make everything decay over time. rewrite this to be delta property?

    integrated_spikes = np.sum(spikes[-500:])
    integrated_input = np.sum(input[-500:])
    exc_rest = V_rest + integrated_input/2000
    exc_th = max(V_rest+5, V_th - integrated_spikes/50)
    exc_refrac = max(par['tau_abs_ref'], tau_ref - integrated_spikes*2.5)
    exc_gain = gain + integrated_spikes*2.5
    # return V_rest, V_th, tau_ref, gain
    return exc_rest, exc_th, exc_refrac, exc_gain


# Excitability function that decays over time. Changes come from spikes. Affects V_rest, V_th (?)
def exc_sigmoid_func(V_rest, V_th, tau_ref, gain, V_m, spikes, input, exc):
    integrated_spikes = np.sum(spikes[-200:])
    num_spikes = np.sum(spikes[-200:]) / par['V_spike']
    integrated_input = np.sum(input[-200:])

    exc_rest = 10 * sigmoid(num_spikes - 20) + V_rest
    # V_rest <= exc_rest <= exc_rest_max
    exc_rest = min(max(exc_rest, V_rest), par['exc_rest_max'])

    exc_th = -5 * sigmoid(num_spikes - 20) + V_th
    # exc_thresh_min <= exc_th <= V_th
    exc_th = min(max(exc_th, par['exc_thresh_min']), V_th)

    # exc_gain = gain + integrated_spikes*2.5
    exc_refrac = -0.002 * sigmoid(num_spikes - 20) + tau_ref
    # tau_abs_ref <= exc_refrac <= tau_ref
    exc_refrac = min(max(exc_refrac, par['tau_abs_ref']), tau_ref)


    # TODO: Change
    exc_gain = gain + num_spikes*2.5

    return exc_rest, exc_th, exc_refrac, exc_gain


def exc_sigmoid_timedep_func(V_rest, V_th, tau_ref, gain, V_m, spikes, input, exc):
    spike_offset = par['spike_offset']
    # Set weights of spikes based on time. Makes sure it is the same size as spikes at this timestep
    if par['timedep_scale'] == 'linear':
        timedep_spikeweights = np.linspace(par['timedep_min_weight'], par['timedep_max_weight'], num = spikes[-par['num_relevant_timebins']:].shape[0])
    elif par['timedep_scale'] == 'geometric' or par['timedep_scale'] == 'logarithmic':
        # geomspace cannot contain 0
        if par['timedep_min_weight'] == 0:
            timedep_spikeweights = np.geomspace(par['timedep_max_weight'], par['timedep_max_weight'] * 2, num = spikes[-par['num_relevant_timebins']:].shape[0]) - par['timedep_max_weight']
        else:
            timedep_spikeweights = np.geomspace(par['timedep_min_weight'], par['timedep_max_weight'], num = spikes[-par['num_relevant_timebins']:].shape[0])

    integrated_spikes = np.sum(spikes[-par['num_relevant_timebins']:])
    num_spikes = np.sum(spikes[-par['num_relevant_timebins']:] * timedep_spikeweights)  / par['V_spike']
    integrated_input = np.sum(input[-par['num_relevant_timebins']:])

    exc_rest = 10 * sigmoid(num_spikes - spike_offset) + V_rest - (exc[0, -1] - V_rest) * dt / tau_exc
    # V_rest <= exc_rest <= exc_rest_max
    exc_rest = min(max(exc_rest, V_rest), par['exc_rest_max'])

    exc_th = -5 * sigmoid(num_spikes - spike_offset) + V_th - (exc[1, -1] - V_th) * dt / tau_exc
    # exc_thresh_min <= exc_th <= V_th
    exc_th = min(max(exc_th, par['exc_thresh_min']), V_th)

    # exc_gain = gain + integrated_spikes*2.5
    exc_refrac = -0.002 * sigmoid(num_spikes - spike_offset) + tau_ref - (exc[2, -1] - tau_ref) * dt / tau_exc
    # tau_abs_ref <= exc_refrac <= tau_ref
    exc_refrac = min(max(exc_refrac, par['tau_abs_ref']), tau_ref)


    # TODO: Change
    exc_gain = gain + num_spikes*2.5 - (exc[3, -1] - gain) * dt / tau_exc

    return exc_rest, exc_th, exc_refrac, exc_gain

def exc_diff_timedep_func(V_rest, V_th, tau_ref, gain, V_m, spikes, input, exc):
    spike_offset = par['spike_offset']
    # Set weights of spikes based on time. Makes sure it is the same size as spikes at this timestep
    if par['timedep_scale'] == 'linear':
        timedep_spikeweights = np.linspace(par['timedep_min_weight'], par['timedep_max_weight'], num = spikes[-par['num_relevant_timebins']:].shape[0])
    elif par['timedep_scale'] == 'geometric' or par['timedep_scale'] == 'logarithmic':
        # geomspace cannot contain 0
        if par['timedep_min_weight'] == 0:
            timedep_spikeweights = np.geomspace(par['timedep_max_weight'], par['timedep_max_weight'] * 2, num = spikes[-par['num_relevant_timebins']:].shape[0]) - par['timedep_max_weight']
        else:
            timedep_spikeweights = np.geomspace(par['timedep_min_weight'], par['timedep_max_weight'], num = spikes[-par['num_relevant_timebins']:].shape[0])

    num_spikes = np.sum(spikes[-par['num_relevant_timebins']:] * timedep_spikeweights)  / par['V_spike']

    exc_rest = 10 * sigmoid(num_spikes - spike_offset) + V_rest - (exc[0, -1] - V_rest) * dt / tau_exc
    # V_rest <= exc_rest <= exc_rest_max
    exc_rest = min(max(exc_rest, V_rest), par['exc_rest_max'])
    exc_rest = exc_rest + np.random.normal(0, 0.5)

    # exc_th = -5 * (np.tanh(num_spikes - spike_offset) + 1) / 2 + V_th - (exc[1, -1] - V_th) * dt / tau_exc
    exc_th = -5 * (num_spikes - spike_offset) / par['num_relevant_timebins'] + V_th - (exc[1, -1] - V_th) * dt / tau_exc
    # exc_thresh_min <= exc_th <= V_th
    exc_th = min(max(exc_th, par['exc_thresh_min']), V_th)
    exc_th = exc_th + np.random.normal(0, 0.25)

    exc_refrac = -0.002 * (math.erf(num_spikes - spike_offset) + 1) / 2 + tau_ref - (exc[2, -1] - tau_ref) * dt / tau_exc
    # tau_abs_ref <= exc_refrac <= tau_ref
    exc_refrac = min(max(exc_refrac, par['tau_abs_ref']), tau_ref)
    exc_refrac = exc_refrac + np.random.normal(0, 0.000125)


    # TODO: Change
    exc_gain = gain + num_spikes*2.5 - (exc[3, -1] - gain) * dt / tau_exc + np.random.normal(0, 1)

    return exc_rest, exc_th, exc_refrac, exc_gain
