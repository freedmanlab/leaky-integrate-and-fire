"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from parameters_lif import par, update_dependencies
# %matplotlib inline

# Parameters:
# TODO: put into parameters

T           = par['T']    # total time to sumulate (msec)
dt          = par['simulation_dt'] # Simulation timestep
time        = par['time']
inpt        = par['inpt']   # Neuron input voltage
neuron_input= par['neuron_input']

num_layers  = par['num_layers']
num_neurons = par['num_neurons']

# Graphing functions:
def plot_neuron_behaviour(time, data, neuron_type, neuron_id, y_title):
    #print ('Drawing graph with time.shape={}, data.shape={}'.format(time.shape, data.shape))
    plt.plot(time,data)
    plt.title('{} @ {}'.format(neuron_type, neuron_id))
    plt.ylabel(y_title)
    plt.xlabel('Time (msec)')
    # Graph to the data with some headroom...
    y_min = 0
    y_max = max(data)*1.2
    if y_max == 0:
        y_max = 1
    plt.ylim([y_min,y_max])
    plt.show()

def plot_membrane_potential(time, Vm, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, Vm, neuron_type, neuron_id, y_title = 'Membrane potential (V)')

def plot_spikes(time, Vm, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, Vm, neuron_type, neuron_id, y_title = 'Spike (V)')

# Basic LIF Neuron class

class LIFNeuron():
    def __init__(self, debug=True):
        # Simulation config (may not all be needed!!)
        self.dt       = par['neuron_dt']       # neuronal time step
        self.t_rest   = par['t_rest']           # initial refractory time

        #LIF Properties
        self.Vm       = np.array([0])    # Neuron potential (mV)
        self.time     = np.array([0])    # Time duration for the neuron (needed?)
        self.spikes   = np.array([0])    # Output (spikes) for the neuron

        self.t        = par['t']         # Neuron time step
        self.Rm       = par['Rm']        # Resistance (kOhm)
        self.Cm       = par['Cm']        # Capacitance (uF)
        self.tau_m    = par['tau_m']     # Time constant
        self.tau_ref  = par['tau_ref']   # refractory period (ms)
        self.Vth      = par['Vth']       # = 1  #spike threshold
        self.V_spike  = par['V_spike']   # spike delta (V)
        self.type     = par['type']
        self.debug    = par['debug']
        if self.debug:
            print ('LIFNeuron(): Created {} neuron starting at time {}'.format(self.type, self.t))

    def spike_generator(self, neuron_input):
        # Create local arrays for this run
        duration = len(neuron_input)
        Vm = np.zeros(duration)  #len(time)) # potential (V) trace over time
        time = np.arange(self.t, self.t+duration)
        spikes = np.zeros(duration)  #len(time))

        if self.debug:
            print ('spike_generator(): Running time period self.t={}, self.t+duration={}'
                   .format(self.t, self.t+duration))

        # Seed the new array with previous value of last run
        Vm[-1] = self.Vm[-1]

        if self.debug:
            print ('LIFNeuron.spike_generator.initial_state(input={}, duration={}, initial Vm={}, t={})'
               .format(neuron_input, duration, Vm[-1], self.t))

        for i in range(duration):
            if self.debug:
                print ('Index {}'.format(i))

            if self.t > self.t_rest:
                Vm[i]=Vm[i-1] + (-Vm[i-1] + neuron_input[i-1]*self.Rm) / self.tau_m * self.dt

                if self.debug:
                    print('spike_generator(): i={}, self.t={}, Vm[i]={}, neuron_input={}, self.Rm={}, self.tau_m * self.dt = {}'
                          .format(i,self.t, Vm[i], neuron_input[i], self.Rm, self.tau_m * self.dt))

                if Vm[i] >= self.Vth:
                    spikes[i] += self.V_spike
                    self.t_rest = self.t + self.tau_ref
                    if self.debug:
                        print ('*** LIFNeuron.spike_generator.spike=(self.t_rest={}, self.t={}, self.tau_ref={})'
                           .format(self.t_rest, self.t, self.tau_ref))

            self.t += self.dt

        # Save state
        self.Vm = np.append(self.Vm, Vm)
        self.spikes = np.append(self.spikes, spikes)
        self.time = np.append(self.time, time)

        if self.debug:
            print ('LIFNeuron.spike_generator.exit_state(Vm={} at iteration i={}, time={})'
                   .format(self.Vm, i, self.t))

# Create neuronal array
def create_neurons(num_layers, num_neurons, debug=False):
    neurons = []
    for layer in range(num_layers):
        if debug:
            print ('create_neurons(): Creating layer {}'.format(layer))
        neuron_layer = []
        for i in range(num_neurons):
            neuron_layer.append(LIFNeuron(debug=debug))
        neurons.append(neuron_layer)
    return neurons

neurons = create_neurons(num_layers, num_neurons, debug=False)
print('Neuronal array shape: {}'.format(np.shape(neurons)))

# Run stimuli for each neuron in layer 0
stimulus_len = len(neuron_input)
layer = 0
for neuron in range(num_neurons):
    offset = random.randint(0,100)   # Simulates stimulus starting at different times
    stimulus = np.zeros_like(neuron_input)
    stimulus[offset:stimulus_len] = neuron_input[0:stimulus_len - offset]
    neurons[layer][neuron].spike_generator(stimulus)

# Graph for neuron in layer 0
plot_membrane_potential(neurons[0][0].time, neurons[0][0].Vm, 'Membrane Potential {}'.format(neurons[0][0].type), neuron_id = "0/0")
plot_spikes(neurons[0][0].time, neurons[0][0].spikes, 'Output spikes for {}'.format(neurons[0][0].type), neuron_id = "0/0")

# Sum spikes for layer 0
layer = 0
layer_spikes = np.zeros_like(neurons[layer][0].spikes)
for i in range(num_neurons):
    layer_spikes += neurons[layer][i].spikes

# Graph spikes for layer 0
plot_spikes(neurons[0][0].time, layer_spikes, 'Output spikes for layer {}'.format(layer))


# Simulate spike propagation through layers
for layer in range(1, num_layers):
    neurons[layer][0].spike_generator(layer_spikes)

    start_time = 0
    end_time = len(neurons[1][0].time)
    print('Rendering neurons[{}][0] over the time period {}:{}'.format(layer,start_time,end_time))

    # Graph results
    plot_spikes(neurons[layer][0].time[start_time:end_time-1], layer_spikes[start_time:end_time],
                'Input Spikes for {}'.format(neurons[layer][0].type), neuron_id = "{}/0".format(layer))
    plot_membrane_potential(neurons[layer][0].time[start_time:end_time], neurons[layer][0].Vm[start_time:end_time],
                'Membrane Potential {}'.format(neurons[layer][0].type), neuron_id = "{}/0".format(layer))
    plot_spikes(neurons[layer][0].time[start_time:end_time], neurons[layer][0].spikes[start_time:end_time],
                'Output spikes for {}'.format(neurons[layer][0].type), neuron_id = "{}/0".format(layer))
