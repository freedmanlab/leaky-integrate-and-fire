"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib
from tqdm import tqdm
from parameters_rlif import par, update_dependencies

T           = par['T']    # total time to simulate (ms)
dt          = par['simulation_dt'] # Simulation timestep
dts        = par['timesteps']
tau_exc  = par['tau_exc']   # exc decay time constant (ms)
current_dts = par['num_current_timesteps']
current_func = par['spikes_to_current_func']
current_profile = current_func(dt*np.arange(current_dts),par['voltage_decay_const'])
input_dts = par['input_timesteps']
V_in       = par['V_in']   # Neuron input voltage
input_stdev = par['input_stdev']
voltage_stdev = par['voltage_stdev']

time_range = np.arange(0, T, dt)

num_neurons = par['num_neurons']
num_inputs = par['num_inputs']
num_exc = par['exc_num']
connections_per_input = par['input_connected_num']//num_inputs

np.random.seed(353)



#builds the connection matrix - each row is a neuron's INPUT
#i.e. connections_ij > 0 -> neuron i connects to neuron j
connections = np.zeros((num_neurons, num_inputs+num_neurons))
rng_matrix = np.random.uniform(size=(num_neurons, num_neurons))
strength_matrix = np.random.uniform(.1, 1, size=(num_neurons, num_neurons))
strength_matrix[rng_matrix > par['recur_connect_frac']] = 0
input_connected_neurons = np.random.choice(num_neurons, par['input_connected_num'], replace = False)
for input_idx in np.arange(num_inputs):
    connections[input_connected_neurons[connections_per_input*input_idx:connections_per_input*(input_idx+1)], input_idx] = np.random.exponential(scale=.5, size=connections_per_input)
connections[:, num_inputs:] = strength_matrix


#makes some neurons inhibitory
inhib_idxs = np.random.choice(num_neurons, num_neurons-num_exc, replace = False)+num_inputs
connections_mult_matrix = np.identity(num_inputs+num_neurons)
connections_mult_matrix[inhib_idxs, inhib_idxs] = -5

connection_matrix = connections@connections_mult_matrix


neuron_input = np.random.normal(0, input_stdev, (num_inputs, dts))
for inpt in np.arange(num_inputs):
    offset = np.random.randint(dts//10, dts // 3)
    neuron_input[inpt, offset:offset + input_dts] += V_in



# Graphing functions:
def plot_neuron_behaviour(time, data, neuron_type, neuron_id, y_title):
    # print ('Drawing graph with time.shape={}, data.shape={}'.format(time.shape, data.shape))
    plt.plot(time, data)
    plt.title('{0} @ {1}'.format(neuron_type, neuron_id))
    plt.ylabel(y_title)
    plt.xlabel('Time (msec)')

    # Graph the data with some headroom
    if min(data) < 0:
        y_min = min(data)*1.2
    elif min(data) == 0:
        y_min = -1
    elif min(data) > 0:
        y_min = min(data)*0.8

    if max(data) < 0:
        y_max = max(data)*0.8
    elif max(data) == 0:
        y_max = 1
    elif max(data) > 0:
        y_max = max(data)*1.2

    plt.ylim([y_min, y_max])
    plt.show()

def plot_membrane_potential(time, V_m, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, V_m, neuron_type, neuron_id, y_title = 'Membrane potential (V)')

def plot_spikes(time, V_m, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, V_m, neuron_type, neuron_id, y_title = 'Spike (V)')

# Basic LIF Neuron class
class LIFNeuron():
    def __init__(self, number, debug=True, **specific_params):
        self.number=number
        # Simulation config (may not all be needed!!)
        self.dt       = par["simulation_dt"]       # neuronal time step
        self.t_rest   = par['t_rest']           # initial refractory time
        #LIF Properties
  # Output (spikes) for the neuron

        self.gain     = par['gain']      # neuron gain (unitless)
        self.t        = 0                # initial time
        self.Rm       = par['Rm']        # Resistance (kOhm)
        self.Cm       = par['Cm']        # Capacitance (uF)
        self.tau_m    = par['tau_m']     # Time constant (ms)
        self.tau_ref  = par['tau_ref']   # refractory period (ms)
        self.tau_exc  = par['tau_exc']   # exc decay time constant (ms)
        self.V_th     = par['V_th']      # = 1  #spike threshold
        self.V_spike  = par['V_spike']   # spike delta (V)
        self.V_rest   = specific_params.get("V_rest", par['V_rest'])    # resting potential (V)
        self.type     = par['type']
        self.debug    = par['debug']
        self.exc_func = specific_params.get("exc_func", par["exc_func"])
        self.connections = connection_matrix[number, :]
        self.input_connected = np.any(self.connections[:num_inputs])

        self.input = np.zeros(dts)
        self.output = np.zeros(dts)
        self.exc      = np.zeros((4,dts))# Resting potential (mV?), threshold (mV?), refractory period (ms), gain (unitless)
        self.V_m      = np.zeros(dts)+self.V_rest    # Neuron potential (mV)
        self.spikes   = np.zeros(dts)
        self.spiketimes = []
        self.exc[0, :] = self.V_rest
        self.exc[1, :] = self.V_th
        self.exc[2, :] = self.tau_ref
        self.exc[3, :] = self.gain
        if self.debug:
            print ('LIFNeuron(): Created {} neuron starting at time {}'.format(self.type, self.t))

    def spike_generator(self, input, timestep):
        # Create local arrays for this run

        if timestep*dt > self.t_rest:
            specific_input = np.sum(input*self.connections)
            self.input[timestep] = specific_input
            self.V_m[timestep] = self.V_m[timestep-1] + np.random.normal(0, voltage_stdev) +\
                (-self.V_m[timestep-1] + self.exc[0,timestep-1] + self.exc[3,timestep-1]*specific_input*self.Rm) / self.tau_m * self.dt
            self.exc[:, timestep] = self.exc_func(self.V_rest, self.V_th, self.tau_ref, self.gain,
                                      self.V_m[:timestep], self.spikes[:timestep], self.input[:timestep], self.exc[:,:timestep])

            if self.V_m[timestep] >= self.exc[1,timestep]:
                self.spikes[timestep] += self.V_spike
                try:
                    self.output[timestep:timestep + current_dts] += current_profile
                except:
                    print("ERROR LELLULULULULU")
                    self.output[timestep:] += current_profile[:dts - timestep]
                self.spiketimes.append(timestep*dt)
                # if i+1 < spikes.shape[0]:
                #     spikes[i+1] += self.V_spike
                self.t_rest = timestep*dt + self.exc[2,timestep]
                if self.debug:
                    print ('*** LIFNeuron.spike_generator.spike=(self.t_rest={}, self.t={}, self.tau_ref={})'.format(self.t_rest, self.t, self.tau_ref))
        else:
            self.exc[:, timestep] = self.exc_func(self.V_rest, self.V_th, self.tau_ref, self.gain,
                            self.V_m[:timestep], self.spikes[:timestep], self.input[:timestep], self.exc[:,:timestep])
            self.V_m[timestep] = self.exc[0,timestep]

# define resting excitability function - params are V_rest, V_m, spikes, I, exc

#make a spike rate increase function and a spike rate decrease function I think
def exc_static_up_func(V_rest, V_th, tau_ref, gain, V_m, spikes, I, exc):
    #make everything decay over time. rewrite this to be delta property?

    integrated_spikes = np.sum(spikes[-500:])
    integrated_current = np.sum(I[-500:])
    exc_rest = V_rest + integrated_current/2000
    exc_th = max(V_rest+.001, V_th - integrated_spikes/50)
    exc_refrac = max(par['tau_abs_ref'], tau_ref - integrated_spikes*2.5)
    exc_gain = gain + integrated_spikes*2.5
    return V_rest, V_th, tau_ref, gain

def exc_static_down_func(V_rest, V_th, tau_ref, gain, V_m, spikes, I, exc):
    return V_rest, V_th, tau_ref, gain

def exc_decay_up_func(V_rest, V_th, tau_ref, gain, V_m, spikes, I, exc):
    exc_rest = exc[0, -1] - (exc[0, -1]-V_rest)*dt/tau_exc
    if spikes[-1] > 0:
        exc_rest += .002
    return exc_rest, V_th, tau_ref, gain

def exc_decay_down_func(V_rest, V_th, tau_ref, gain, V_m, spikes, I, exc):
    exc_rest = exc[0, -1] - (exc[0, -1]-V_rest)*dt/tau_exc
    if spikes[-1] > 0:
        exc_rest -= .005
    return exc_rest, V_th, tau_ref, gain

# Create neuronal array
def add_neurons(num_neurons, neuron_list=[], debug=False, **specific_params):
    num_prev_neurons = len(neuron_list)
    for i in range(num_neurons):
        neuron_list.append(LIFNeuron(i+num_prev_neurons, debug=debug, **specific_params))
    return neuron_list

neurons_up = add_neurons(num_neurons//2, [], False, exc_func = exc_decay_up_func)
neurons = add_neurons(num_neurons-num_neurons//2, neurons_up, debug=False, exc_func = exc_decay_down_func)
num_input_connected_neurons = np.sum([neurons[neuron].input_connected for neuron in np.arange(num_neurons)])

full_input = np.zeros((num_inputs+num_neurons,dts))
full_input[:num_inputs, :] = neuron_input
for timestep in tqdm(np.arange(1,dts)):
    for neuron in np.arange(num_neurons):
        full_input[num_inputs+neuron, timestep] = neurons[neuron].output[timestep-1]
    for neuron in np.arange(num_neurons):
        neurons[neuron].spike_generator(full_input[:, timestep], timestep)

full_output = np.zeros((num_neurons, dts))
for neuron in np.arange(num_neurons):
    full_output[neuron, :] = neurons[neuron].V_m

fig, axs = plt.subplots(2,1, sharex=True)
plt.get_current_fig_manager().window.showMaximized()
axs[0].set_xlim([0, T])
for input_num in np.arange(num_inputs):
    axs[0].plot(time_range, full_input[input_num, :], "b,")
axs[0].axhline(par['V_th'], color="r")
for output_num in np.arange(num_neurons):
    axs[0].plot(time_range, neurons[output_num].V_m, ",")

neuron_spiketimes = [neurons[neuron].spiketimes for neuron in np.arange(num_neurons)]
neuron_input_connections = [neurons[neuron].input_connected for neuron in np.arange(num_neurons)]
sorted_neuron_spiketimes = [spiketime for spiketime, tf in sorted(zip(neuron_spiketimes, neuron_input_connections), key=lambda neuron: neuron[1], reverse=True)]
neuron_colors = ["b"] * num_neurons
neuron_colors[:num_input_connected_neurons] = ["r"]*num_input_connected_neurons
axs[1].eventplot(sorted_neuron_spiketimes, colors=neuron_colors)
axs[0].set_title("input")
axs[1].set_title("output")

fig2, axs2 = plt.subplots(2,1, sharex=True)
plt.get_current_fig_manager().window.showMaximized()
for neuron_num in np.arange(num_neurons):
    input_sum = np.sum(neurons[neuron_num].input)
    print("{0}: {1}".format(neurons[neuron_num].input_connected, input_sum))
    if neurons[neuron_num].input_connected:
        axs2[0].plot(time_range, neurons[neuron_num].input, linewidth=1)
    else:
        axs2[1].plot(time_range, neurons[neuron_num].input, linewidth=1)
axs2[0].set_title("input connected")
axs2[1].set_title("input not connected")

fig3, axs3 = plt.subplots(4,1, sharex=True)
for exc_prop in np.arange(4):
    for exc_neuron in np.arange(50):
        axs3[exc_prop].plot(time_range, neurons[exc_neuron].exc[exc_prop,:], "b")
    axs3[exc_prop].plot(time_range, neurons[55].exc[exc_prop, :], "r")
    axs3[exc_prop].plot(time_range, neurons[60].exc[exc_prop, :], "r")

fig3.suptitle("Excitability properties")
plt.get_current_fig_manager().window.showMaximized()
plt.show()


