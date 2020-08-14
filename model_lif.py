"""
Antony Simonoff 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
from parameters_lif import par, update_dependencies
from analysis import plot_neuron_behaviour, plot_membrane_potential, plot_spikes
from scipy.stats import norm

"""
CURRENT COMPLEXITY: O(n^3) DUE TO 3 NESTED FOR LOOPS IN SPIKE PROPAGATION SIMULATION.
COULD NEED TO BE REFACTORED.
"""

# Set seeds for reproduction
random.seed(42)
np.random.seed(42)

T           = par['T']    # total time to sumulate (msec)
dt          = par['simulation_dt'] # Simulation timestep
time        = par['time']
inpt        = par['inpt']   # Neuron input voltage
# neuron_input= np.zeros(par['neuron_input'].shape)
neuron_input= par['neuron_input']
neuron_input[500:2000] = par['inpt'] * 1.5

num_layers  = par['num_layers']
num_neurons = par['num_neurons']
num_input_neurons = par['num_input_neurons']
neuron_connections = par['neuron_connections']

synaptic_plasticity = par['synaptic_plasticity']

# Basic LIF Neuron class
class LIFNeuron():
    def __init__(self, debug=True, **specific_params):
        # Simulation config (may not all be needed!)
        self.dt       = dt       # neuronal time step. Equal to simulation time step
        self.t        = specific_params.get("t", 0) # start time, can vary for different neurons
        self.t_rest_absolute= specific_params.get("t_rest_absolute", par["t_rest_absolute"]) #absolute refractory time
        self.t_rest_relative= specific_params.get("t_rest_relative", par["t_rest_relative"]) #relative refractory time

        #LIF Properties
        self.exc      = np.zeros((4,1))  # Resting potential (mV?), threshold (mV?), refractory period (ms), gain (unitless)
        self.V_m      = np.zeros(1)    # Neuron potential (mV)
        self.time     = np.zeros(1)    # Time duration for the neuron (needed?)
        self.spikes   = np.zeros(1)    # Output (spikes) for the neuron
        self.spiketimes=[]             # Time of when spikes happen

        self.gain     = specific_params.get("gain", par["gain"])      # neuron gain (unitless)
        self.Rm       = specific_params.get("Rm", par["Rm"])        # Resistance (kOhm)
        self.Cm       = specific_params.get("Cm", par["Cm"])        # Capacitance (uF)
        self.tau_m    = specific_params.get("tau_m", par["tau_m"])     # Time constant (ms)
        self.tau_ref  = specific_params.get("tau_ref", par["tau_ref"])   # refractory period (ms)
        self.V_th     = specific_params.get("V_th", par["V_th"])       # spike threshold (mV)
        self.V_spike  = specific_params.get("V_spike", par["V_spike"])   # spike delta (mV)
        self.V_rest   = specific_params.get("V_rest", par['V_rest'])    # resting potential (mV)
        self.V_hyperpolar = specific_params.get("V_hyperpolar", par["V_hyperpolar"]) # hyperpolarization potential (mV)
        self.type     = specific_params.get("type", par["type"])
        self.debug    = specific_params.get("debug", par["debug"])
        self.exc_func = specific_params.get("exc_func", par["exc_func"])

        self.synaptic_plasticity = specific_params.get("synaptic_plasticity", par["synaptic_plasticity"])
        self.syn_plas_constant = specific_params.get("syn_plas_constant", par["syn_plas_constant"])

        if self.synaptic_plasticity:
            self.n_std_devs = specific_params.get("n_std_devs", par["n_std_devs"])
            loc = np.floor(num_neurons/2)
            scale = np.ceil(num_neurons/self.n_std_devs)/2

            self.neuron_number = specific_params.get("neuron_number")

            self.synaptic_plasticity = []

            for neuron in range(num_neurons):
                cdf = norm.cdf(neuron, loc = loc, scale = scale) * 2
                if cdf > 1:
                    cdf = 1 - cdf%1

                cdf = cdf * self.syn_plas_constant

                self.synaptic_plasticity.append(cdf)

            self.synaptic_plasticity = np.roll(self.synaptic_plasticity, int(num_neurons/2))
            self.synaptic_plasticity = np.roll(self.synaptic_plasticity, specific_params.get("neuron_number"))


        if self.debug:
            print ('LIFNeuron(): Created {} neuron starting at time {}'.format(self.type, self.t))

    def spike_generator(self, neuron_input):
        # Create local arrays for this run
        duration = len(neuron_input)
        V_m = np.full(duration, self.V_rest) #len(time)) # potential (mV) trace over time
        exc = np.full((4, duration), self.V_rest)

        # TODO: change this to be consistent throughout code
        self.tau_ref = self.t_rest_absolute

        exc[1, :] = self.V_th
        exc[2, :] = self.tau_m
        exc[3, :] = self.gain
        # exc[4, :] = self.V_hyperpolar
        time = np.arange(self.t, self.t+duration)
        spikes = np.zeros(duration)  #len(time))

        # make sure inputs are spikes
        for neuron in neuron_input:
            if neuron >= 0:
                neuron = self.V_spike

        if self.debug:
            print ('spike_generator(): Running time period self.t={}, self.t+duration={}'
                   .format(self.t, self.t+duration))

        if self.debug:
            print ('LIFNeuron.spike_generator.initial_state(input={}, duration={}, initial V_m={}, t={})'
               .format(neuron_input, duration, V_m[-1], self.t))

        for i in range(duration):
            if self.debug:
                print ('Index {}'.format(i))

            if self.t > self.t_rest_absolute + self.t_rest_relative:

                V_m[i]=V_m[i-1] + (-V_m[i-1] + exc[0,i-1] + exc[3,i-1]*neuron_input[i-1]*self.Rm) / self.tau_m * self.dt
                exc[:, i] = self.exc_func(self.V_rest, self.V_th, self.tau_ref, self.gain,
                                          V_m[:i], spikes[:i], neuron_input[:i], exc[:,:i])
                if self.debug:
                    print('spike_generator(): i={}, self.t={}, V_m[i]={}, neuron_input={}, self.Rm={}, self.tau_m * self.dt = {}'
                          .format(i,self.t, V_m[i], neuron_input[i], self.Rm, self.tau_m * self.dt))

                if V_m[i] >= exc[1,i]:
                    spikes[i] += self.V_spike
                    self.t_rest_absolute = self.t + exc[2,i]
                    V_m[i] = self.V_spike
                    self.spiketimes.append(self.t)

                    if self.debug:
                        print ('*** LIFNeuron.spike_generator.spike=(self.t_rest_absolute={}, self.t={}, self.tau_ref={})'.format(self.t_rest_absolute, self.t, self.tau_ref))

            elif self.t_rest_absolute < self.t < self.t_rest_absolute + self.t_rest_relative:
                exc[:, i] = self.exc_func(self.V_rest, self.V_th, self.tau_ref, self.gain,
                                V_m[:i], spikes[:i], neuron_input[:i], exc[:,:i])

                # V_m[i] = exc[4,i]
                V_m[i] = self.V_hyperpolar # TODO: Make decay be biological
            else:
                exc[:, i] = self.exc_func(self.V_rest, self.V_th, self.tau_ref, self.gain,
                                V_m[:i], spikes[:i], neuron_input[:i], exc[:,:i])
                V_m[i] = exc[0,i]

            self.t += self.dt

        # Save state
        # if the first time spikes are being generated
        if self.spikes.shape[0] == 1:
            self.spikes = spikes
            self.exc = exc
            self.V_m = V_m
            self.time = time
        else:
            self.exc = np.hstack((self.exc, exc))
            self.V_m = np.append(self.V_m, V_m)
            self.spikes = np.append(self.spikes, spikes)
            self.time = np.append(self.time, time)

        if self.debug:
            print ('LIFNeuron.spike_generator.exit_state(V_m={} at iteration i={}, time={})'
                   .format(self.V_m, i, self.t))

# define resting excitability function - params are V_rest, V_m, spikes, I, exc
def exc_func(V_rest, V_th, tau_ref, gain, V_m, spikes, I, exc):
    integrated_spikes = np.sum(spikes[-500:])
    integrated_current = np.sum(I[-500:])
    exc_rest = V_rest + integrated_spikes/10
    exc_thresh = V_th - integrated_current/2500
    exc_refrac = max(0.1, tau_ref - integrated_spikes*1.5)
    exc_gain = gain + integrated_spikes/2
    return V_rest, exc_thresh, tau_ref, exc_gain

# Create neuronal array
def create_neurons(num_layers, num_neurons, debug=False, **specific_params):
    print("Making neuronal array...")
    neurons = []
    for layer in range(num_layers):
        if debug:
            print ('create_neurons(): Creating layer {}'.format(layer))
        neuron_layer = []
        for i in range(num_neurons):
            specific_params = {'neuron_number': i}
            neuron_layer.append(LIFNeuron(debug=debug, **specific_params))
        neurons.append(neuron_layer)
    print("Finished neuronal array creation")
    return neurons

neurons = create_neurons(num_layers, num_neurons, debug = False, exc_func = exc_func)

"""Calculate spike propagation through layers"""
layer_spikes = []

# Calculate inputs
for layer in np.arange(num_layers):
    if layer == 0: # Sum spikes in layer 0
        # Run stimuli for each neuron in layer 0
        stimulus_len = len(neuron_input)
        for neuron in np.arange(num_neurons):
            stimulus = np.zeros_like(neuron_input)
            # TODO: Make this be not random and encode task information
            # TODO: Create input encoding function, pass results to layer 0 neurons
            indices = np.random.choice(np.arange(stimulus_len), random.randrange(len(stimulus)), replace=False)
            for i in indices:
                stimulus[i] = inpt
            neurons[layer][neuron].spike_generator(stimulus)

        # Break up for loops so neurons[..].spikes is created
        layer_spikes.append(np.zeros_like(neurons[layer][0].spikes))
        for neuron in np.arange(num_neurons):
            layer_spikes[layer] += neurons[layer][neuron].spikes

    else:
        layer_spikes.append(np.zeros_like(neurons[layer-1][0].spikes))
        neuron_start = int(np.ceil(-neuron_connections/2))
        neuron_end = int(np.ceil(neuron_connections/2))

        for neuron in np.arange(num_neurons):
            input_spikes = np.zeros_like(neurons[layer-1][0].spikes)

            if synaptic_plasticity:
                for input_neuron in range(num_neurons):
                    connection_strength = neurons[layer][neuron].synaptic_plasticity[input_neuron]
                    input_spikes += neurons[layer-1][input_neuron].spikes * connection_strength
            else:
                for i in range(neuron_start, neuron_end):
                    # par['neuron_connections'] project to this neuron
                    input_spikes += neurons[layer-1][(neuron+i)%num_neurons].spikes

            neurons[layer][neuron].spike_generator(input_spikes)
            layer_spikes[layer] += neurons[layer][neuron].spikes
        print("b")
        print(layer_spikes[layer])
        print(np.shape(layer_spikes[layer]))
        # input()

    start_time = 0
    end_time = time
    print('Rendering neurons[{}][0] over the time period {}:{} ms'.format(layer, start_time, end_time))

    """Graph results:"""
    # Raster plots:
    fig, axs = plt.subplots(2,1)
    fig.suptitle("Rendering neurons, Rm = {}, Cm = {}, tau_m = {}".format(neurons[layer][0].Rm, neurons[layer][0].Cm, neurons[layer][0].tau_m))
    # for input_num in np.arange(num_inputs):
    #     axs[0].plot(time_range, full_input[input_num, :], "b,")
    # for output_num in np.arange(num_neurons):
    #     axs[0].plot(time_range, neurons[0][neuron].V_m, "r,")
    axs[1].eventplot([neurons[layer][neuron].spiketimes for neuron in np.arange(num_neurons)])
    axs[0].set_title("input")
    axs[1].set_title("output")

    plt.show()

    plot_spikes(neurons[layer][0].time[start_time:end_time], layer_spikes[layer][start_time:end_time],
    'Input Spikes for {}'.format(neurons[layer][0].type), neuron_id = "{}/0".format(layer))
    plot_membrane_potential(neurons[layer][0].time[start_time:end_time], neurons[layer][0].V_m[start_time:end_time],
    'Membrane Potential {}'.format(neurons[layer][0].type), neuron_id = "{}/0".format(layer))
    plot_spikes(neurons[layer][0].time[start_time:end_time], neurons[layer][0].spikes[start_time:end_time],
    'Output spikes for {}'.format(neurons[layer][0].type), neuron_id = "{}/0".format(layer))
