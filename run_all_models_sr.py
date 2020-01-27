import numpy as np
from parameters_sr import *
import model_sr as model
import sys


def try_model(gpu_id):
    try:
        # Run model
        model.main(gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

# Wrapped into main
if __name__ == "__main__":

    # Handle args
    try:
        gpu_id = sys.argv[1]
        print('Selecting GPU ', gpu_id)
    except:
        gpu_id = None

    # Update parameters
    update_parameters({ 'simulation_reps'           : 0,
                        'batch_train_size'          : 1024,
                        'learning_rate'             : 0.002,
                        'noise_rnn_sd'              : 0.5,
                        'noise_in_sd'               : 0.1,
                        'num_iterations'            : 3000,
                        'spike_regularization'      : 'L1', # L1 for large-scale tests
                        'synapse_config'            : 'full',
                        'test_cost_multiplier'      : 3,
                        'balance_EI'                : True,
                        'weight_cost'               : 1.0,
                        'spike_cost'                : 1e-2,
                        'fix_time'                  : 500,
                        'sample_time'               : 650,
                        'delay_time'                : 1000,
                        'test_time'                 : 650,
                        'savedir'                   : './savedir/',
                        'iters_between_outputs'     : 1})

    # set up sequence reproduction task
    task_list = ['SR']

    # Run models
    for task in task_list:
        update_parameters({'trial_type'     : task,
                           'excitability'   : False})
        try_model(gpu_id)
