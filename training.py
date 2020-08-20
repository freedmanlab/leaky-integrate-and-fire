import numpy as np
from model_lif import exc_func

from parameters_lif import par

def create_neurons(num_layers = par['num_layers'], num_neurons = par['num_neurons'], debug=False, **specific_params):
    neurons = []
    for layer in range(num_layers):
        if debug:
            print ('create_neurons(): Creating layer {}'.format(layer))
        neuron_layer = []
        for i in range(num_neurons):
            specific_params = {'neuron_number': i}
            neuron_layer.append(LIFNeuron(debug=debug, **specific_params))
        neurons.append(neuron_layer)
    return neurons

def encode_task(num_input_neurons = par['num_input_neurons'], task_info = None):
    # TEMP:
    task_info = np.arange(num_input_neurons)
    input_layer = create_neurons(num_layers = 1, num_neurons = num_input_neurons, debug = False, exc_func = exc_func)
    input_layer = input_layer[0]
    for input_neuron in np.arange(num_input_neurons):
        input_layer[input_neuron].spikes = np.zeros_like(neuron_input)
        split = int(len(neuron_input)/num_input_neurons)
        input_layer[input_neuron].spikes[split * input_neuron:split * (input_neuron + 1)] = par['V_spike'] # change V_spike to something more accurate
    return input_layer
