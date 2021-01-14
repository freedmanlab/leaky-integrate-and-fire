import celluloid
import numpy as np
import matplotlib.pyplot as plt

def animate(neurons, dts, num_neurons):
    # timesteps = 10
    timesteps = dts

    heatmap = np.zeros((timesteps, 3, 2, 10, 10))
    spikes = np.zeros((timesteps, 100))

    exc_labels = ["Spikes", "Null", "Resting Voltage (mV)", "Threshold Voltage (mV)", "Refractory Time (ms)", "Gain"]
    offsets = [V_rest, V_th, par['tau_ref'], 0]

    fig, axs = plt.subplots(3, 2)
    camera = celluloid.Camera(fig)

    for timestep in tqdm(np.arange(1, timesteps), desc = "Rendering"): # dts
        spikes[timestep] = [neurons[0][neuron].spikes[timestep] / V_spike for neuron in np.arange(num_neurons)]

        for row in np.arange(np.shape(axs)[0]):
            for col in np.arange(np.shape(axs)[1]):
                for i in np.arange(10):
                    for j in np.arange(10):
                        if row == 0 and col == 0:
                            heatmap[timestep][row][col][i][j] = spikes[timestep][i * 10 + j]
                        elif row != 0:
                            heatmap[timestep][row][col][i][j] = neurons[0][i * 10 + j].exc[row * 2 + col - 2, timestep] - offsets[row * 2 + col - 2]

                axs[row][col].imshow(heatmap[timestep][row][col], cmap = 'hot', interpolation = 'nearest')
                axs[row][col].set_title(exc_labels[row * 2 + col])
        fig.suptitle('Frame {}'.format(timestep))
        camera.snap()

    animation = camera.animate()
    animation.save('animations/animation {}.gif'.format(time.time()))
