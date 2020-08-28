"""
Antony Simonoff 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
from analysis import plot_neuron_behaviour, plot_membrane_potential, plot_spikes
from scipy.stats import norm

from parameters_lif import par, update_dependencies
from training import poisson_spikes

"""
CURRENT COMPLEXITY: O(n^3) DUE TO 3 NESTED FOR LOOPS IN SPIKE PROPAGATION SIMULATION.
COULD NEED TO BE REFACTORED.
"""

update_dependencies()
print("--> Parameters loaded successfully")

# TODO: Change all main matrices to one big matrix

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

n_hidden  = par['n_hidden']
num_neurons = par['num_neurons']
num_input_neurons = par['num_input_neurons']
num_output_neurons = par['num_output_neurons']
neuron_connections = par['neuron_connections']
excitability_ratio = par['excitability_ratio']

synaptic_plasticity = par['synaptic_plasticity']

task_info = par['task_info']

debug = par['debug']

# Initialize all global matrices
synaptic_weights = np.zeros((num_layers, num_neurons, num_neurons)) # shape = layer, this neuron, projected neuron

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

        self.baseline_fr= specific_params.get("baseline_fr", par["baseline_fr"])
        self.preferred_angle_fr = specific_params.get("preferred_angle_fr", par["preferred_angle_fr"])

        self.synaptic_plasticity = specific_params.get("synaptic_plasticity", par["synaptic_plasticity"])
        self.syn_plas_constant = specific_params.get("syn_plas_constant", par["syn_plas_constant"])

        if self.synaptic_plasticity:
            self.n_std_devs = specific_params.get("n_std_devs", par["n_std_devs"])
            loc = np.floor(num_neurons/2)
            scale = np.ceil(num_neurons/self.n_std_devs)/2

            self.layer_number = specific_params.get("layer_number")
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
        for spike in neuron_input:
            if spike >= 0:
                spike = self.V_spike

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
def create_neurons(n_hidden = par['n_hidden'], num_neurons = par['num_neurons'], debug=False, excitability_ratio = par['excitability_ratio'], **specific_params):
    neurons = []
    for layer in range(n_hidden):
        inhibitory_neurons = np.random.choice(np.arange(num_neurons), int(num_neurons * (1 - excitability_ratio)))
        if debug:
            print ('create_neurons(): Creating layer {}'.format(layer))
        neuron_layer = []
        for i in range(num_neurons):
            specific_params = {'layer_number': i, 'neuron_number': i}
            neuron_layer.append(LIFNeuron(debug=debug, **specific_params))
        for inhibitory_neuron in inhibitory_neurons:
            for neuron in range(num_neurons):
                neuron_layer[neuron].synaptic_plasticity[inhibitory_neuron] = neuron_layer[neuron].synaptic_plasticity[inhibitory_neuron] * -1
        neurons.append(neuron_layer)
    return neurons

def encode_task(num_input_neurons = par['num_input_neurons'], task_info = task_info):
    if task_info == 'DMS':
        input_layer = create_neurons(n_hidden = 1, num_neurons = num_input_neurons, excitability_ratio = 1,  debug = False, exc_func = exc_func)[0]
        match = random.randrange(8)
        p = np.full(8, (1-par['p_match_sample_eq'])/7)
        p[match] = par['p_match_sample_eq']
        sample = np.random.choice(np.arange(8), p = p)
        # sample = random.randrange(8) # TODO: Change to weighh match higher

        for input_neuron in np.arange(num_input_neurons):
            input_layer[input_neuron].spikes = np.zeros_like(neuron_input)

            """ DMS task encoding """
            # Fixation:
            spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].baseline_fr, return_n_bin = True)
            input_layer[input_neuron].spikes[0:500] = spikes
            for bin in bins_n:
                input_layer[input_neuron].spiketimes.append((bin + 0) * dt)

            # Sample
            if input_neuron == match:
                spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].preferred_angle_fr, return_n_bin = True)
                input_layer[input_neuron].spikes[500:1000] = spikes
                for bin in bins_n:
                    input_layer[input_neuron].spiketimes.append((bin + 500) * dt)
            elif input_neuron == (match + 1) % num_input_neurons or input_neuron == (match - 1) % num_input_neurons:
                spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].preferred_angle_fr * 0.6, return_n_bin = True)
                input_layer[input_neuron].spikes[500:1000] = spikes
                for bin in bins_n:
                    input_layer[input_neuron].spiketimes.append((bin + 500) * dt)
            else:
                spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].baseline_fr, return_n_bin = True)
                input_layer[input_neuron].spikes[500:1000] = spikes
                for bin in bins_n:
                    input_layer[input_neuron].spiketimes.append((bin + 500) * dt)

            # Delay
            spikes, bins_n = poisson_spikes(n_bins = 1000, fr = input_layer[input_neuron].baseline_fr, return_n_bin = True)
            input_layer[input_neuron].spikes[1000:2000] = spikes
            for bin in bins_n:
                input_layer[input_neuron].spiketimes.append((bin + 1000) * dt)

            # Sample
            if input_neuron == sample:
                spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].preferred_angle_fr, return_n_bin = True)
                input_layer[input_neuron].spikes[2000:2500] = spikes
                for bin in bins_n:
                    input_layer[input_neuron].spiketimes.append((bin + 2000) * dt)
            elif input_neuron == (sample + 1) % num_input_neurons or input_neuron == (sample - 1) % num_input_neurons:
                spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].preferred_angle_fr * 0.6, return_n_bin = True)
                input_layer[input_neuron].spikes[2000:2500] = spikes
                for bin in bins_n:
                    input_layer[input_neuron].spiketimes.append((bin + 2000) * dt)
            else:
                spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].baseline_fr, return_n_bin = True)
                input_layer[input_neuron].spikes[2000:2500] = spikes
                for bin in bins_n:
                    input_layer[input_neuron].spiketimes.append((bin + 2000) * dt)

            # Output
            spikes, bins_n = poisson_spikes(n_bins = 500, fr = input_layer[input_neuron].baseline_fr, return_n_bin = True)
            input_layer[input_neuron].spikes[2500:3000] = spikes
            for bin in bins_n:
                input_layer[input_neuron].spiketimes.append((bin + 2500) * dt)

        return input_layer, match, sample

    elif task_info == 'baseline_firing':
        # TEMP:
        input_layer = create_neurons(n_hidden = 1, num_neurons = num_input_neurons, excitability_ratio = 1, debug = False, exc_func = exc_func)[0]
        for input_neuron in np.arange(num_input_neurons):
            input_layer[input_neuron].spikes = np.zeros_like(neuron_input)
            for spike in np.arange(len(neuron_input)):
                if spike % input_layer[input_neuron].baseline_fr == 0:
                    input_layer[input_neuron].spikes[spike] = par['V_spike']
        return input_layer, match, sample

    else:
        task_info = np.arange(num_input_neurons)
        input_layer = create_neurons(n_hidden = 1, num_neurons = num_input_neurons, excitability_ratio = 1, debug = False, exc_func = exc_func)[0]
        split = int(len(neuron_input)/num_input_neurons)
        for input_neuron in np.arange(num_input_neurons):
            input_layer[input_neuron].spikes = poisson_spikes(n_bins = len(neuron_input))
        return input_layer



def main():
    """ Main loop function """
    print("--> Encoding task")
    if task_info == 'DMS':
        input_layer, match, sample = encode_task()
    else:
        input_layer = encode_task()

    print("--> Finished task encoding")

    print("--> Making neuronal array")
    neurons = create_neurons(n_hidden, num_neurons, debug = False, exc_func = exc_func)
    output_layer = create_neurons(1, num_output_neurons, debug = False, excitability_ratio = 1, exc_func = exc_func)[0]
    print("--> Finished neuronal array creation")

    """Calculate spike propagation through layers"""
    layer_spikes = []

    task_accuracy = []

    # Calculate inputs
    for layer in np.arange(n_hidden + 1): # Propagates through hidden layers and then output layer
        if layer == 0: # Sum spikes in layer 0
            split = int(num_neurons / num_input_neurons)
            for input_neuron in np.arange(num_input_neurons):
                for neuron in np.arange(split * input_neuron, split * (input_neuron + 1)):
                    neurons[layer][neuron].spike_generator(input_layer[input_neuron].spikes)

            # Break up for loops so neurons[..].spikes is created
            layer_spikes.append(np.zeros_like(neurons[layer][0].spikes))
            for neuron in np.arange(num_neurons):
                layer_spikes[layer] += neurons[layer][neuron].spikes

        elif 0 < layer < n_hidden:
            layer_spikes.append(np.zeros_like(neurons[layer-1][0].spikes))

            for neuron in np.arange(num_neurons):
                input_spikes = np.zeros_like(neurons[layer-1][0].spikes)

                if synaptic_plasticity:
                    for input_neuron in range(num_neurons):
                        connection_strength = neurons[layer][neuron].synaptic_plasticity[input_neuron]
                        input_spikes += neurons[layer-1][input_neuron].spikes * connection_strength
                else:
                    for i in range(neuron_start, neuron_end):
                        # par['neuron_connections'] project to this neuron
                        neuron_start = int(np.ceil(-neuron_connections/2))
                        neuron_end = int(np.ceil(neuron_connections/2))
                        input_spikes += neurons[layer-1][(neuron+i)%num_neurons].spikes

                neurons[layer][neuron].spike_generator(input_spikes)
                layer_spikes[layer] += neurons[layer][neuron].spikes

        else:
            split = int(num_neurons / num_output_neurons)
            layer_spikes.append(np.zeros_like(neurons[layer-1][0].spikes))
            for output_neuron in np.arange(num_output_neurons):
                for neuron in np.arange(split * output_neuron, split * (output_neuron+1)):
                    input_spikes = np.zeros_like(neurons[layer-1][0].spikes)
                    input_spikes += neurons[layer-1][neuron].spikes

                output_layer[output_neuron].spike_generator(input_spikes)
                layer_spikes[layer] += output_layer[output_neuron].spikes


        start_time = 0
        end_time = time
        print('Propagating through layer {} over the time period {}:{} ms'.format(layer, start_time, end_time))

        """Graph results:"""
        # Raster plots:
        if 0 < layer < n_hidden and False: # Change for graphing
            fig, axs = plt.subplots(2,1)
            fig.suptitle("Rendering neurons, layer {}, Rm = {}, Cm = {}, tau_ref = {}".format(layer, neurons[layer][0].Rm, neurons[layer][0].Cm, neurons[layer][0].tau_m))
            axs[0].eventplot([neurons[layer-1][neuron].spiketimes for neuron in np.arange(num_neurons)])
            axs[1].eventplot([neurons[layer][neuron].spiketimes for neuron in np.arange(num_neurons)])
            axs[0].set_title("Layer {} spikes".format(layer-1))
            axs[1].set_title("Layer {} spikes".format(layer))

            plt.show()

        elif layer == n_hidden and True: # Change for graphing
            fig, axs = plt.subplots(2,1)
            fig.suptitle("Input and Output, Rm = {}, Cm = {}, tau_ref = {}".format(neurons[0][0].Rm, neurons[0][0].Cm, neurons[0][0].tau_m))
            # axs[0].eventplot([neurons[0][neuron].spiketimes for neuron in np.arange(num_neurons)], linewidths=1.0)
            axs[0].eventplot([input_layer[input_neuron].spiketimes for input_neuron in np.arange(num_input_neurons)], linewidths=1.0)
            axs[1].eventplot([output_layer[output_neuron].spiketimes for output_neuron in np.arange(num_output_neurons)], linewidths=1.0)
            axs[0].set_title("input")
            # axs[0].set_xlabel("time, s")
            axs[0].set_ylabel("neuron")
            axs[1].set_title("output")
            axs[1].set_xlabel("time, s")
            axs[1].set_ylabel("neuron")

            plt.show()

        if layer == 0 or layer == n_hidden -1 : # Change for graphing
            plot_membrane_potential(neurons[layer][0].time[start_time:end_time], neurons[layer][0].V_m[start_time:end_time],
            'Membrane Potential {}'.format(neurons[layer][0].type), neuron_id = "Layer = {}, neuron = {}".format(layer, 0))

    def decode_task(output_layer = output_layer, task_info = task_info):
        if task_info == "DMS":
            task_error = []
            targets = [par['preferred_angle_fr'], par['baseline_fr']]

            if match == sample:
                neuron_1_error = targets[0] - sum(output_layer[0].spikes[2500:3000]/(0.5 * par['V_spike']))
                neuron_2_error = targets[1] - sum(output_layer[1].spikes[2500:3000]/(0.5 * par['V_spike']))
                task_error = np.average([neuron_1_error, neuron_2_error])
                trials.append("match")
            if match != sample:
                neuron_1_error = targets[1] - sum(output_layer[0].spikes[2500:3000]/(0.5 * par['V_spike']))
                neuron_2_error = targets[0] - sum(output_layer[1].spikes[2500:3000]/(0.5 * par['V_spike']))
                task_error = np.average([neuron_1_error, neuron_2_error])
                trials.append("nonmatch")

            task_accuracy.append(task_error)

        else:
            pass

    print("--> Decoding task")
    trials = []
    decode_task()
    print("Task error:", task_accuracy, trials)
    print("--> Completed task decoding")

main()
