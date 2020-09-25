import numpy as np

from parameters_rlif_AS import par
from utils_rlif_AS import sigmoid

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
    # Set weights of spikes based on time. Makes sure it is the same size as spikes at this timestep
    timedep_spikeweights = np.arange(spikes[-par['num_relevant_timebins']:].shape[0]) / (spikes[-par['num_relevant_timebins']:].shape[0] / 2)


    integrated_spikes = np.sum(spikes[-par['num_relevant_timebins']:])
    num_spikes = np.sum(spikes[-par['num_relevant_timebins']:] * timedep_spikeweights)  / par['V_spike']
    integrated_input = np.sum(input[-par['num_relevant_timebins']:])

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
