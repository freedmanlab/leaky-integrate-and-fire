"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd
import random
import matplotlib
from tqdm import tqdm
from parameters_rlif import par, update_dependencies

T           = par['T']    # total time to simulate (msec)
dt          = par['simulation_dt'] # Simulation timestep
dts        = par['timesteps']
input_dts = par['input_timesteps']
V_in       = par['V_in']   # Neuron input voltage

time_range = np.arange(0, T, dt)

num_layers  = par['num_layers']
num_neurons = par['num_neurons']
num_inputs = par['num_inputs']

np.random.seed(3)

neuron_input = np.random.normal(0, .005, (num_inputs, dts))
for inpt in np.arange(num_inputs):
    offset = np.random.randint(0, dts / 10)
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


#generation of connectivity arrays
def generate_connections(par):
    connections = np.zeros(num_inputs+num_neurons)
    yn_in = np.random.uniform()
    if yn_in < par["input_connect_binary_prob"]:
        c_probs = np.random.uniform(size=num_inputs)
        for input in np.arange(c_probs.shape[0]):
            if c_probs[input] < par["input_connect_frac"]:
                connections[input] = np.random.uniform(.9,1.1)
    c_probs = np.random.uniform(size=num_neurons)
    for neuron in np.arange(c_probs.shape[0]):
        if c_probs[neuron] < par["recur_connect_frac"]:
            connections[neuron+num_inputs] = np.random.uniform(.9, 1.1)
    return connections

# Basic LIF Neuron class
class LIFNeuron():
    def __init__(self, debug=True, **specific_params):
        # Simulation config (may not all be needed!!)
        self.dt       = par["simulation_dt"]       # neuronal time step
        self.t_rest   = par['t_rest']           # initial refractory time

        #LIF Properties
  # Output (spikes) for the neuron

        self.gain     = par['gain']      # neuron gain (unitless)
        self.t        = par['t_initial']         # initial time
        self.Rm       = par['Rm']        # Resistance (kOhm)
        self.Cm       = par['Cm']        # Capacitance (uF)
        self.tau_m    = par['tau_m']     # Time constant (ms)
        self.tau_ref  = par['tau_ref']   # refractory period (ms)
        self.V_th     = par['V_th']      # = 1  #spike threshold
        self.V_spike  = par['V_spike']   # spike delta (V)
        self.V_rest   = specific_params.get("V_rest", par['V_rest'])    # resting potential (V)
        self.type     = par['type']
        self.debug    = par['debug']
        self.exc_func = specific_params.get("exc_func", par["exc_func"])
        self.connections = generate_connections(par)

        self.input = np.zeros(dts)
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
            self.V_m[timestep] = self.V_m[timestep-1] + np.random.normal(0, .005) +\
                (-self.V_m[timestep-1] + self.exc[0,timestep-1] + self.exc[3,timestep-1]*specific_input*self.Rm) / self.tau_m * self.dt
            self.exc[:, timestep] = self.exc_func(self.V_rest, self.V_th, self.tau_ref, self.gain,
                                      self.V_m[:timestep], self.spikes[:timestep], self.input[:timestep], self.exc[:,:timestep])

            if self.V_m[timestep] >= self.exc[1,timestep]:
                self.spikes[timestep] += self.V_spike
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
def exc_func(V_rest, V_th, tau_ref, gain, V_m, spikes, I, exc):
    integrated_spikes = np.sum(spikes[-500:])
    integrated_current = np.sum(I[-500:])
    exc_rest = V_rest + integrated_spikes/10
    exc_thresh = V_th - integrated_current/2500
    exc_refrac = max(1, tau_ref - integrated_spikes*2.5)
    exc_gain = gain + integrated_spikes/2
    return V_rest, V_th, tau_ref, gain

# Create neuronal array
def create_neurons(num_layers, num_neurons, debug=False, **specific_params):
    neurons = []
    for layer in range(num_layers):
        if debug:
            print ('create_neurons(): Creating layer {}'.format(layer))
        neuron_layer = []
        for i in range(num_neurons):
            neuron_layer.append(LIFNeuron(debug=debug, **specific_params))
        neurons.append(neuron_layer)
    return neurons

neurons = create_neurons(num_layers, num_neurons, debug=False, exc_func = exc_func)

full_input = np.zeros((num_inputs+num_neurons,dts))
full_input[:num_inputs, :] = neuron_input
for timestep in tqdm(np.arange(1,dts)):
    for neuron in np.arange(num_neurons):
        full_input[num_inputs+neuron, timestep] = neurons[0][neuron].spikes[timestep-1]
    for neuron in np.arange(num_neurons):
        neurons[0][neuron].spike_generator(full_input[:, timestep], timestep)

full_output = np.zeros((num_neurons, dts))
for neuron in np.arange(num_neurons):
    full_output[neuron, :] = neurons[0][neuron].V_m

fig, axs = plt.subplots(2,1)
for input_num in np.arange(num_inputs):
    axs[0].plot(time_range, full_input[input_num, :], "b,")
for output_num in np.arange(num_neurons):
    axs[0].plot(time_range, neurons[0][neuron].V_m, "r,")
axs[1].eventplot([neurons[0][neuron].spiketimes for neuron in np.arange(num_neurons)])
axs[0].set_title("input")
axs[1].set_title("output")

plt.show()


