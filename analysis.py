import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from parameters_rlif_AS import par
from tqdm import tqdm

import numpy as np

def plot_neuron_behaviour(time, data, neuron_type, neuron_id, y_title):
    global figsize
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

def plot_spikepulse():
    figsize = [9.5, 5]

    x = np.linspace(0, 100, 100) * 0.001
    tau = 0.005
    y = 0.0101*(np.exp(-(x/(4*tau)))-np.exp(-(x/tau)))
    y[y<0.0005] = 0
    figsize = [9.5, 5]

    fig, axs = plt.subplots(1, 1, sharex = True, figsize = figsize)

    axs.plot(x, y)
    # axs.set_title("Average Firing Rate of Input Connected Neuron Inputs (NICs)", fontsize = 20)
    axs.set_ylabel("Output Voltage, mV", fontsize = 16)
    # axs.set_ylabel("Firing Rate, Hz", fontsize = 16)
    axs.set_title("Output Spikepulse", fontsize = 20)
    axs.set_xlabel("Time, s", fontsize = 16)
    axs.tick_params(axis = 'both', labelsize = 12)
    axs.tick_params(axis = 'both', labelsize = 12)

    plt.show()

def plot_fr(neurons, num_neurons, num_input_connected_neurons, dts, sliding_window = 25):
    figsize = [9.5, 5]
    spike_counts_nic = np.zeros((num_input_connected_neurons, dts - sliding_window)) # Input connected SMA
    spike_counts_nnc = np.zeros((num_neurons - num_input_connected_neurons, dts - sliding_window)) # Input not connected SMA

    for timestep in tqdm(np.arange(dts - sliding_window), desc = "generating SMAs"):
        x = 0
        y = 0
        for neuron in np.arange(num_neurons):
            if neurons[0][neuron].input_connected:
                spike_counts_nic[x][timestep] = np.count_nonzero(neurons[0][neuron].spikes[timestep:timestep+ sliding_window])
                x += 1
            else:
                spike_counts_nnc[y][timestep] = np.count_nonzero(neurons[0][neuron].spikes[timestep:timestep+ sliding_window])
                y += 1

    spike_counts_nic_avg = np.mean(spike_counts_nic, axis = 0) / 0.025
    spike_counts_nnc_avg = np.mean(spike_counts_nnc, axis = 0) / 0.025

    fig, axs = plt.subplots(2, 1, sharex = True, figsize = figsize)
    axs[0].plot(spike_counts_nic_avg)
    axs[1].plot(spike_counts_nnc_avg)
    # axs[0].set_ylim(axs[1].get_ylim())
    axs[0].set_title("Average NIC Firing Rate", fontsize = 20)
    axs[0].set_ylabel("Firing Rate, Hz", fontsize = 16)
    axs[1].set_ylabel("Firing Rate, Hz", fontsize = 16)
    axs[1].set_title("Average NNC Firing Rate", fontsize = 20)
    axs[1].set_xlabel("Time, ms", fontsize = 16)
    axs[0].tick_params(axis = 'both', labelsize = 12)
    axs[1].tick_params(axis = 'both', labelsize = 12)
    plt.show()

def plot_smas(neurons, num_neurons, num_input_connected_neurons, dts, sliding_window = 25):
    figsize = [9.5, 5]
    sma_ic = np.zeros((num_input_connected_neurons, dts - sliding_window)) # Input connected SMA
    sma_nic = np.zeros((num_neurons - num_input_connected_neurons, dts - sliding_window)) # Input not connected SMA

    for timestep in tqdm(np.arange(dts - sliding_window), desc = "generating SMAs"):
        x = 0
        y = 0
        for neuron in np.arange(num_neurons):
            if neurons[0][neuron].input_connected:
                sma_ic[x][timestep] = np.average(neurons[0][neuron].input[timestep:timestep+ sliding_window])
                x += 1
            else:
                sma_nic[y][timestep] = np.average(neurons[0][neuron].input[timestep:timestep+ sliding_window])
                y += 1

    sma_ic_avg = np.mean(sma_ic, axis = 0)
    sma_nic_avg = np.mean(sma_nic, axis = 0)

    fig1, axs1 = plt.subplots(2, 1, sharex = True, figsize = figsize)
    axs1[0].plot(sma_ic_avg)
    axs1[1].plot(sma_nic_avg)

    axs1[0].set_title("SMA of NIC Inputs", fontsize = 20)
    axs1[0].set_ylabel("\u0394mV", fontsize = 16)
    axs1[1].set_ylabel("\u0394mV", fontsize = 16)
    axs1[1].set_title("SMA of NNC Inputs", fontsize = 20)
    axs1[1].set_xlabel("Time, ms", fontsize = 16)
    axs1[0].tick_params(axis = 'both', labelsize = 12)
    axs1[1].tick_params(axis = 'both', labelsize = 12)
    plt.show()

def plot_fano(neurons, num_neurons, num_input_connected_neurons, dts, sliding_window = 50):
    figsize = [9.5, 5]
    spike_counts_nic = np.zeros((num_input_connected_neurons, dts - sliding_window)) # Input connected SMA
    spike_counts_nnc = np.zeros((num_neurons - num_input_connected_neurons, dts - sliding_window)) # Input not connected SMA

    for timestep in tqdm(np.arange(dts - sliding_window), desc = "generating Fano Factors"):
        x = 0
        y = 0
        for neuron in np.arange(num_neurons):
            if neurons[0][neuron].input_connected:
                spike_counts_nic[x][timestep] = np.count_nonzero(neurons[0][neuron].spikes[timestep:timestep+ sliding_window])
                x += 1
            else:
                spike_counts_nnc[y][timestep] = np.count_nonzero(neurons[0][neuron].spikes[timestep:timestep+ sliding_window])
                y += 1

    spike_counts_nic_avg = np.mean(spike_counts_nic, axis = 0)
    spike_counts_nnc_avg = np.mean(spike_counts_nnc, axis = 0)

    spike_counts_nic_avg = np.mean(spike_counts_nic, axis = 0)
    spike_counts_nnc_avg = np.mean(spike_counts_nnc, axis = 0)
    spike_counts_nic_var = np.var(spike_counts_nic, axis = 0)
    spike_counts_nnc_var = np.var(spike_counts_nic, axis = 0)
    fano_factor_nic = spike_counts_nic_var / spike_counts_nic_avg
    fano_factor_nnc = spike_counts_nnc_var / spike_counts_nnc_avg

    fig, axs = plt.subplots(2, 1, sharex = True, figsize = figsize)
    axs[0].plot(fano_factor_nic)
    axs[1].plot(fano_factor_nnc)
    axs[0].set_ylim(axs[1].get_ylim())
    axs[0].set_title("Average NIC Fano Factor", fontsize = 20)
    axs[0].set_ylabel("Fanor Factor", fontsize = 16)
    axs[1].set_ylabel("Fanor Factor", fontsize = 16)
    axs[1].set_title("Average NNC Fano Factor", fontsize = 20)
    axs[1].set_xlabel("Time, ms", fontsize = 16)
    axs[0].tick_params(axis = 'both', labelsize = 12)
    axs[1].tick_params(axis = 'both', labelsize = 12)
    plt.show()

def plot_exc_props(neurons, time_range, sorted_neuron_spiketimes, V_rest, V_th, graphed_neuron):
    figsize = [9.5, 5]
    fig3, axs3 = plt.subplots(4,1, sharex=True, figsize = figsize)
    exc_labels = ["Resting Voltage ($V_{rest}$)", "Threshold Voltage ($V_{th}$)", "Refractory Time ($\\tau_{ref}$)", "Gain ($\Gamma$)"]
    y_labels = ["mV", "mV", "sec", None]
    for exc_prop in np.arange(4):
        axs3[exc_prop].plot(time_range, neurons[0][graphed_neuron].exc[exc_prop, :])
        axs3[exc_prop].set_title(exc_labels[exc_prop], fontsize = 20)
        axs3[exc_prop].set_ylabel(y_labels[exc_prop], fontsize = 16)
    axs3[0].set_ylim([V_rest - 2, par['exc_rest_max'] + 2])
    axs3[1].set_ylim([par['exc_thresh_min'] - 2, V_th + 2])
    axs3[2].set_ylim([par['tau_abs_ref'] - 0.002, par['tau_ref'] + 0.002])
    axs3[3].set_xlabel("Time, s", fontsize = 16)
    fig3.suptitle('Excitability Properties of Neuron {}, {} spikes'.format(graphed_neuron, np.shape(sorted_neuron_spiketimes[graphed_neuron])[0]), fontsize = 24)

    plt.show()

def plot_isis(neurons, graphed_neuron):
    figsize = [9.5, 5]
    isis = []
    neuron = graphed_neuron
    print(len(neurons[0][neuron].spiketimes))
    for spiketime in np.arange(1, len(neurons[0][neuron].spiketimes)):
        isis.append(neurons[0][neuron].spiketimes[spiketime] - neurons[0][neuron].spiketimes[spiketime-1])

    fig, axs = plt.subplots(1, 1, figsize = figsize)
    axs.set_ylabel("Number of Spikes", fontsize = 16)
    axs.set_xlabel("Interspike Interval, s", fontsize = 16)
    axs.hist(isis, bins = 50)
    axs.set_title("Interspike Intervals of Neuron {}, {} spikes".format(neuron, len(neurons[0][neuron].spiketimes)), fontsize = 20)
    axs.set_xlim((0, 0.2))
    axs.tick_params(axis = 'both', labelsize = 12)
    axs.tick_params(axis = 'both', labelsize = 12)
    plt.show()

def plot_sorted_spikes(neuron_spiketimes, neurotransmitters, num_neurons, sorted_neuron_spiketimes, num_input_connected_neurons, graphed_neuron):
    figsize = [9.5, 5]
    fig5, axs5 = plt.subplots(2, 1, sharex=True, figsize = figsize)
    sorted_neurotransmitter_spiketimes = [spiketime for spiketime, tf in sorted(zip(neuron_spiketimes, neurotransmitters), key=lambda neuron: neuron[1], reverse=True)]
    num_AMPA = np.count_nonzero(neurotransmitters == "AMPA")
    num_NMDA = np.count_nonzero(neurotransmitters == "NMDA")
    num_GABA = np.count_nonzero(neurotransmitters == "GABA")
    neurotransmitter_colors = ['y'] * num_neurons
    neurotransmitter_colors[:num_AMPA] = ['b'] * num_AMPA
    neurotransmitter_colors[num_AMPA:num_NMDA + num_AMPA] = ['g'] * num_NMDA
    neurotransmitter_colors[num_NMDA + num_AMPA:] = ['y'] * num_GABA
    print(num_AMPA, num_NMDA, num_GABA)

    axs5[0].eventplot(sorted_neurotransmitter_spiketimes, colors = neurotransmitter_colors)
    AMPA_patch = mpatches.Patch(color='blue', label='AMPA neurons')
    NMDA_patch = mpatches.Patch(color='green', label='NMDA neurons')
    GABA_patch = mpatches.Patch(color='yellow', label='GABA neurons')
    axs5[0].legend(handles = [GABA_patch, NMDA_patch, AMPA_patch], loc = 'upper right')

    neuron_colors = ['b'] * num_neurons
    neuron_colors[:num_input_connected_neurons] = ['r']*num_input_connected_neurons
    neuron_colors[graphed_neuron] = 'lime'

    axs5[1].set_xlabel("Time, s", fontsize = 16)
    axs5[1].eventplot(sorted_neuron_spiketimes, colors=neuron_colors)
    input_connected_patch = mpatches.Patch(color='red', label='Input Connected')
    input_not_connected_patch = mpatches.Patch(color='blue', label='Input Not Connected')
    graphed_neuron_patch = mpatches.Patch(color='lime', label='Graphed Neuron')
    axs5[1].legend(handles = [input_not_connected_patch, input_connected_patch, graphed_neuron_patch], loc = 'upper right')


    axs5[0].set_title("Output Spikes per Neuron, Neurotransmitters", fontsize = 20)
    axs5[0].set_ylabel("Neuron", fontsize = 16)
    axs5[1].set_ylabel("Neuron", fontsize = 16)
    axs5[1].set_title("Output Spikes per Neuron, Input Connected", fontsize = 20)
    axs5[1].set_xlabel("Time, s", fontsize = 16)
    axs5[0].tick_params(axis = 'both', labelsize = 12)
    axs5[1].tick_params(axis = 'both', labelsize = 12)
    fig5.suptitle('Sorted Neuronal Spikes', fontsize = 24)
    axs5[0].tick_params(axis = 'both', labelsize = 12)
    axs5[1].tick_params(axis = 'both', labelsize = 12)

    plt.show()

def plot_misc_graphs(neurons, time_range, num_inputs, graphed_neuron, sorted_neuron_spiketimes, full_input, num_neurons, neuron_input_connections, num_input_connected_neurons, dts):
    figsize = [9.5, 5]
    fig, axs = plt.subplots(2,1, sharex=True)
    for input_num in np.arange(num_inputs):
        axs[0].plot(time_range, full_input[input_num, :], 'b,')
    axs[0].axhline(par['V_th'], color='r')
    for output_num in np.arange(num_neurons):
        axs[0].plot(time_range, neurons[0][output_num].V_m, ',')

    graphed_neuron = np.random.choice(np.arange(num_input_connected_neurons, num_neurons))
    graphed_neuron = 44
    neuron_spiketimes = [neurons[0][neuron].spiketimes for neuron in np.arange(num_neurons)]
    neuron_input_connections = [neurons[0][neuron].input_connected for neuron in np.arange(num_neurons)]
    sorted_neuron_spiketimes = [spiketime for spiketime, tf in sorted(zip(neuron_spiketimes, neuron_input_connections), key=lambda neuron: neuron[1], reverse=True)]

    tau_ref_min = [min(neurons[0][neuron].exc[2, :]) for neuron in np.arange(num_neurons)]

    neuron_colors = ['b'] * num_neurons
    neuron_colors[:num_input_connected_neurons] = ['r']*num_input_connected_neurons
    neuron_colors[graphed_neuron] = 'lime'
    axs[1].eventplot(sorted_neuron_spiketimes, colors=neuron_colors)
    axs[0].set_ylabel("Input Voltage (mV)")
    axs[1].set_ylabel("Output Spikes Per Neuron")
    axs[1].set_xlabel("Time, s")
    axs[0].set_title('Input')
    axs[1].set_title('Output')

    axs[0].set_title("Input", fontsize = 20)
    axs[0].set_ylabel("Input Voltage (mV)", fontsize = 16)
    axs[1].set_ylabel("Firing Rate, Hz", fontsize = 16)
    axs[1].set_title("Output", fontsize = 20)
    axs[1].set_xlabel("Time, ms", fontsize = 16)
    axs[0].tick_params(axis = 'both', labelsize = 12)
    axs[1].tick_params(axis = 'both', labelsize = 12)

    fig2, axs2 = plt.subplots(2,1, sharex=True)
    for neuron_num in np.arange(num_neurons):
        if neurons[0][neuron_num].input_connected:
            axs2[0].plot(time_range, neurons[0][neuron_num].input, linewidth=1)
        else:
            axs2[1].plot(time_range, neurons[0][neuron_num].input, linewidth=1)
    axs2[0].set_ylabel("\u0394mV")
    axs2[1].set_ylabel("\u0394mV")
    axs2[1].set_xlabel("Time, s")
    axs2[0].set_title('Input connected')
    axs2[1].set_title('Input not connected')

    fig4, axs4 = plt.subplots(4, 1, sharex=True)
    exc_avg = np.zeros((4, dts))
    exc_labels = ["Resting Voltage ($V_{rest}$)", "Threshold Voltage ($V_{th}$)", "Refractory Time ($\\tau_{ref}$)", "Gain ($\Gamma$)"]
    for exc_prop in np.arange(4):
        exc_avg[exc_prop, :] = np.mean([neuron.exc[exc_prop, :] for neuron in neurons[0]])
        axs4[exc_prop].plot(time_range, exc_avg[exc_prop, :])
        axs4[exc_prop].set_title(exc_labels[exc_prop])
    axs4[3].set_xlabel("Time, s")
    fig4.suptitle('Average Network Excitability Properties')

    fig6, axs6 = plt.subplots(1, 1, sharex=True)
    axs6.plot(time_range, neurons[0][graphed_neuron].V_m)
    fig6.suptitle('Membrane Potential of Neuron {}'.format(graphed_neuron))

    plt.show()
