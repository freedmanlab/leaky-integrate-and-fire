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

def excitability_synaptic(V_rest, V_th, tau_ref, gain, V_m, spikes, input, exc, neurotransmitter, iteration):
    """
    Notes for this simulation of synaptic plasticity and excitability:
    Can simulate a bunch of channels. Which ones to simulate?
    Also need to add in noises
    Options:
        Coming from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1679792/:
        NMDA, AMPA, kainate, nAch, GABA, VGCC (L, P/Q, R types), BK, SK

        Important ones:
            NMDA, AMPA (excitatory); GABA (inhibitory)

        AMPA:
            "Signalling through the AMPA receptor is dynamically modulated by two principal mechanisms: direct phosphorylation of receptor subunits, and changes in the density of receptors at the postsynaptic membrane."
            "Phosphorylation increases the open probability of the receptor in LTP, whereas dephosphorylation is induced during LTD. The concentration of AMPA receptors at the synapse increases after the induction of LTP, whereas it drops during LTD"

        NMDA: (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3482462/)
            "Several groups have noted that NMDAR-LTP can develop over a longer timescale relative to AMPAR-LTP. At unitary connections between layer 5 pyramidal neurons in visual cortical slices, early LTP of the AMPAR-mediated component is followed by a delayed NMDAR-LTP, which seems to restore the AMPAR-to-NMDAR ratio"
            "NMDARs also carry a substantial fraction of the total synaptic charge and may be important for recurrent excitation in cortical networks."

        GABA: (https://onlinelibrary.wiley.com/doi/full/10.1111/jnc.13742)
            "Short‐term plasticity of inhibitory synapses has been associated to changes in the probability of presynaptic release or to alteration in the postsynaptic conductance through receptor desensitization."
            "Thus, iLTP was associated with the increase in GABARAP‐dependent GABAAR exocytosis, which promotes the accumulation of receptors on the postsynaptic membrane, as observed in vitro and in vivo"

    """

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

    if neurotransmitter == 'AMPA':
        # AMPA Simulation:
        # Two parts: increasing open probability of receptor; and increasing density of receptors at membrane
        # First is faster (smaller spike_offset), second is slower (larger spike_offset(?))

        # Increasing open probability of receptor = increasing V_rest, decreasing V_th, increasing gain
        exc_th = -2.5 * sigmoid(num_spikes - par['AMPA_spikeoffset_short']) + V_th - (exc[1, -1] - V_th) * dt / tau_exc

        exc_gain = gain + num_spikes*2.5 - (exc[3, -1] - gain) * dt / tau_exc

        # Increasing density of receptors at membrane = increasing V_rest, decreasing V_th, increasing gain (twice as powerful). Slight exc_refrac changes
        exc_rest = 8 * sigmoid(num_spikes - par['AMPA_spikeoffset_long']) + V_rest - (exc[0, -1] - V_rest) * dt / par['tau_exc_long']

        exc_th = -5 * sigmoid(num_spikes - par['AMPA_spikeoffset_long']) + V_th - (exc[1, -1] - V_th) * dt / par['tau_exc_long']

        exc_refrac = -0.001 * sigmoid(num_spikes - par['AMPA_spikeoffset_long']) + tau_ref - (exc[2, -1] - tau_ref) * dt / par['tau_exc_long']

        exc_gain = gain + num_spikes*1 - (exc[3, -1] - gain) * dt / par['tau_exc_long']
    elif neurotransmitter == 'NMDA':
        # NMDA Simulation:
        # Same as long AMPA, but longer time offset. Large impacts on gain
        exc_rest = 10 * sigmoid(num_spikes - par['NMDA_offset_long']) + V_rest - (exc[0, -1] - V_rest) * dt / par['tau_exc_long']

        exc_th = -5 * sigmoid(num_spikes - par['NMDA_offset_long']) + V_th - (exc[1, -1] - V_th) * dt / par['tau_exc_long']

        exc_refrac = -0.002 * sigmoid(num_spikes - par['NMDA_offset_long']) + tau_ref - (exc[2, -1] - tau_ref) * dt / par['tau_exc_long']

        exc_gain = gain + num_spikes*3 - (exc[3, -1] - gain) * dt / par['tau_exc_long']
    elif neurotransmitter == 'GABA':
        # GABA Simulation:
        # Same as AMPA, but inhibitory

        # Increasing open probability of receptor = increasing V_rest, decreasing V_th, increasing gain
        exc_th = -2.5 * sigmoid(num_spikes - par['GABA_spikeoffset_short']) + V_th - (exc[1, -1] - V_th) * dt / tau_exc

        exc_gain = gain + num_spikes*2.5 - (exc[3, -1] - gain) * dt / tau_exc

        # Increasing density of receptors at membrane = increasing V_rest, decreasing V_th, increasing gain (twice as powerful). Slight exc_refrac changes
        exc_rest = 8 * sigmoid(num_spikes - par['GABA_spikeoffset_long']) + V_rest - (exc[0, -1] - V_rest) * dt / par['tau_exc_long']

        exc_th = -5 * sigmoid(num_spikes - par['GABA_spikeoffset_long']) + V_th - (exc[1, -1] - V_th) * dt / par['tau_exc_long']

        exc_refrac = -0.001 * sigmoid(num_spikes - par['GABA_spikeoffset_long']) + tau_ref - (exc[2, -1] - tau_ref) * dt / par['tau_exc_long']

        exc_gain = gain + num_spikes*1 - (exc[3, -1] - gain) * dt / par['tau_exc_long']


    exc_rest = min(max(exc_rest, V_rest), par['exc_rest_max'])
    exc_th = min(max(exc_th, par['exc_thresh_min']), V_th)
#     exc_refrac = min(max(exc_refrac, par['tau_abs_ref']), tau_ref)

#     # Add natural membrane noise
#     exc_rest = exc_rest + np.random.normal(0, 0.5)
# #     exc_rest = V_rest + np.random.normal(0, 0.5)
#     exc_th = exc_th + np.random.normal(0, 0.25)
# #     exc_th = V_th + np.random.normal(0, 0.25)
#     exc_refrac = exc_refrac + np.random.normal(0, 0.000125)
# #     exc_refrac = tau_ref + np.random.normal(0, 0.000125)
#     exc_gain = exc_gain + np.random.normal(0, 1.0)
# #     exc_gain = gain + np.random.normal(0, 1.0)

    # Add natural membrane noise
    if iteration % 2  == 0:
        exc_rest = exc_rest + np.random.normal(0, 0.5)
    else:
        exc_rest = V_rest + np.random.normal(0, 0.5)

    if iteration % 4 < 2:
        exc_th = exc_th + np.random.normal(0, 0.25)
    else:
        exc_th = V_th + np.random.normal(0, 0.25)

    if iteration % 8 < 4:
        exc_refrac = exc_refrac + np.random.normal(0, 0.000125)
    else:
        exc_refrac = tau_ref + np.random.normal(0, 0.000125)

    if iteration < 8:
        exc_gain = exc_gain + np.random.normal(0, 1.0)
    else:
        exc_gain = gain + np.random.normal(0, 1.0)

    """exc_rest = V_rest
    exc_th = V_th
    exc_refrac = tau_ref
    exc_gain = gain"""

    return exc_rest, exc_th, exc_refrac, exc_gain
