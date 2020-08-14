import matplotlib
import matplotlib.pyplot as plt

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
    plot_neuron_behaviour(time, V_m, neuron_type, neuron_id, y_title = 'Membrane potential (mV)')

def plot_spikes(time, V_m, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, V_m, neuron_type, neuron_id, y_title = 'Spike (mV)')
