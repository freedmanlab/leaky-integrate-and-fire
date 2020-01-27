"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import analysis
import pickle
import time
from parameters_sr import par, update_trial_params, update_dependencies, update_parameters
import os, sys
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")


class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=0)
        self.target_data = target_data
        self.mask = mask

        self.initialize_weights()

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    def initialize_weights(self):
        # Initialize all weights, biases, and initial values

        self.var_dict = {}
        pretrained_names = set(['Wf', 'Wi', 'Wc', 'Wo', 'Uf', 'Ui', 'Uc', 'Uo', 'bf', 'bi', 'bc', 'bo', 'w_out', 'b_out'])
        if par['pretrained_demonstrator']:
            pretrained_vals = pickle.load((f"{par['save_dir']}LSTM_weights_SR.pkl", 'rb'))

        # all keys in par with a suffix of '0' are initial values of trainable variables
        for k, v in par.items():
            if k[-1] == '0':
                name = k[:-1]
                if name in pretrained_names and par['pretrained_demonstrator']:
                    self.var_dict[name] = tf.Variable(pretrained_vals[name], name=name, trainable=False)
                else:
                    self.var_dict[name] = tf.Variable(par[k], name=name)

        self.syn_x_init = tf.constant(par['syn_x_init'])
        self.syn_u_init = tf.constant(par['syn_u_init'])
        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            self.w_rnn = tf.constant(par['EI_matrix']) @ tf.nn.relu(self.var_dict['w_rnn'])
        else:
            self.w_rnn  = self.var_dict['w_rnn']

        # Make reset operations
        reset_vars_ops = []
        for k, v in self.var_dict.items():
            name = k + "0"
            reset_vars_ops.append(tf.assign(v, par[name]))
        self.reset_vars_ops = tf.group(*reset_vars_ops)
        self.resaturation_op = tf.assign(self.var_dict['w_rnn'], par['new_init'])

    def run_model(self):

        self.h = []
        self.syn_x = []
        self.syn_u = []
        self.y = []

        h = self.var_dict['h']
        syn_x = self.syn_x_init
        syn_u = self.syn_u_init

        # Production module records
        self.h_lstm = []
        self.c_lstm = []
        h_lstm, c_lstm, input_lstm = tf.zeros_like(h), tf.zeros_like(h), tf.zeros_like(h)

        # Loop through inputs to memory module, compute state updates
        for t, rnn_input in enumerate(self.input_data):

            # Compute next state + facilitation values
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)

            # Buffer results for future storage
            self.h.append(h)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)

            # If at the last timepoint before the sequence reproduction begins, save hidden state activity
            # in memory module and pipe that through as input to the LSTM network for the rest of the trial
            if t == par['last_seed_timept']:
                input_lstm = h

            h_lstm, c_lstm = self.lstm(input_lstm, h_lstm, c_lstm)
            self.h_lstm.append(h_lstm)
            self.c_lstm.append(c_lstm)

            self.y.append(h_lstm @ tf.nn.relu(self.var_dict['w_out']) + self.var_dict['b_out'])

        # Stack outputs and return
        self.h        = tf.stack(self.h)
        self.syn_x    = tf.stack(self.syn_x)
        self.syn_u    = tf.stack(self.syn_u)
        self.y        = tf.stack(self.y)
        self.h_lstm   = tf.stack(self.h_lstm)
        self.c_lstm   = tf.stack(self.c_lstm)
        self.y_output = tf.nn.softmax(self.y)


    def lstm(self, rnn_input, h, c):
        # Update neural activity and short-term synaptic plasticity values
        #print_dict = {'h': h, 'rnn_input': rnn_input, 'Wf':self.var_dict['Wf'], 'Uf': self.var_dict['Uf'], 'bf': self.var_dict['bf']}
        #for k, v in print_dict.items():
        #    print(k, v.shape)
        f  = tf.sigmoid(rnn_input @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
        i  = tf.sigmoid(rnn_input @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
        cn = tf.tanh(rnn_input @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
        c  = f * c + i * cn
        o  = tf.sigmoid(rnn_input @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])

        # Compute hidden state
        h = o * tf.tanh(c)

        return h, c

    def rnn_cell(self, rnn_input, h, syn_x, syn_u):
        # Update neural activity and short-term synaptic plasticity values

        # Update the synaptic plasticity paramaters
        if par['synapse_config'] is not 'none':#None:
            # implement both synaptic short term facilitation and depression
            syn_x += (par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h)*par['dynamic_synapse']
            syn_u += (par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h)*par['dynamic_synapse']
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h

        else:
            # no synaptic plasticity
            h_post = h

        ### ADD HERE!!!!!!!!! FOR EXCITABILITY CHANGES OF ALL KINDS!

        # Update the hidden state. Only use excitatory projections from input layer to RNN
        # All input and RNN activity will be non-negative
        if par['excitability']:
            h = tf.nn.relu(h * (1-par['alpha_neuron']) \
                + par['alpha_neuron'] * par['exc_states']*(rnn_input @ tf.nn.relu(self.var_dict['w_in']) \
                + h_post @ self.w_rnn + self.var_dict['b_rnn']) \
                + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))
        else:
            h = tf.nn.relu(h * (1-par['alpha_neuron']) \
                + par['alpha_neuron'] * (rnn_input @ tf.nn.relu(self.var_dict['w_in']) \
                + h_post @ self.w_rnn + self.var_dict['b_rnn']) \
                + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))

        return h, syn_x, syn_u


    def optimize(self):

        # Calculate the loss functions and optimize the weights
        self.perf_loss = tf.reduce_mean(self.mask*tf.nn.softmax_cross_entropy_with_logits_v2(\
            logits = self.y, labels = self.target_data, axis = 2))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if par['spike_regularization'] == 'L2' else 1
        self.spike_loss = tf.reduce_mean(self.h**n)
        self.weight_loss = tf.reduce_mean(tf.nn.relu(self.w_rnn)**n)

        self.loss = self.perf_loss + par['spike_cost']*self.spike_loss + par['weight_cost']*self.weight_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        # Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        for grad, var in grads_and_vars:
            if 'w_rnn' in var.op.name:
                grad *= par['w_rnn_mask']
            elif 'w_out' in var.op.name:
                grad *= par['w_out_mask']
            elif 'w_in' in var.op.name:
                grad *= par['w_in_mask']
            capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_op = opt.apply_gradients(capped_gvs)
        self.reset_opt = tf.variables_initializer(opt.variables())

def update_memory_bank(memory_bank_id, perf):

    memory_bank_id += 1
    par['memory_bank_id'] = memory_bank_id
    update_dependencies()
    i = 0
    if memory_bank_id > par['n_memory_banks']:
        pickle.dump(perf, open("memory_bank_performance.pkl", 'wb'))
        i = par['num_iterations']
    print(f"\n\nBeginning memory bank {memory_bank_id}.")

    return memory_bank_id, i

def main(gpu_id = None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])


    # Print key parameters
    print_important_params()

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    # Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()

    # Define all placeholder
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
    x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'input')
    t = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'target')

    # Make sure savedir exists
    if not os.path.exists(f'savedir/{gpu_id}/'):
        os.makedirs(f'savedir/{gpu_id}/')

    save_increment = 0

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=tf.ConfigProto()) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'

        t0 = time.time()
        with tf.device(device):
            model = Model(x, t, m)
        print(f"Model initialized. Time elapsed: {str(time.time() - t0)}")
        print(par['shape'])

        sess.run(tf.global_variables_initializer())

        # keep track of the model performance across training
        t0 = time.time()

        # Set up records for memory bank performance storage
        memory_bank_performance = dict()
        for k in range(par['n_memory_banks']):
            memory_bank_performance[k] = []


        memory_bank_id = 0

        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
        'weight_loss': [], 'iteration': []}

        i = 0
        while i < par['num_iterations']:

            # Generate batch of batch_train_size
            trial_info = stim.generate_trial(set_rule = None)

            # Run the model
            _, loss, perf_loss, spike_loss, weight_loss, y, h, syn_x, syn_u = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, \
                model.weight_loss, model.y, model.h, model.syn_x, model.syn_u], \
                {x: trial_info['neural_input'], t: trial_info['desired_output'], m: trial_info['train_mask']})

            accuracies = analysis.get_perf_sr(trial_info['desired_output'], y, trial_info['train_mask'])

            model_performance = append_model_performance(model_performance, accuracies, loss, perf_loss, spike_loss, weight_loss, i)

            if (np.mean(model_performance['accuracy'][-50:])) > 0.90 or (i == par['num_iterations'] - 1):
                memory_bank_id, i = update_memory_bank(memory_bank_id, memory_bank_performance)
                old_w = sess.run(model.var_dict)['w_rnn']
                old_w[old_w <= 1e-5] = 1e-3
                par['new_init'] = old_w
                _ = sess.run(model.resaturation_op)


            # Save the network model and output model performance to screen
            elif (i+1)%par['iters_between_outputs']==0:
                t1 = time.time()
                #accuracies = [analysis.get_perf_sr(trial_info['desired_output'], y[n,:,:,:], trial_info['train_mask'], True) for n in range(par['n_networks'])]
                print_results(i, perf_loss, spike_loss, weight_loss, h, accuracies)
                print(f"Elapsed time: {str(t1 - t0)}")
                t0 = time.time()

                #memory_bank_performance[memory_bank_id].append(accuracies)

                # Also: compute accuracies on each of the previous memory banks
                # (IDEA HERE: SHOULD WE CONSTRAIN S.T. NO INDIVIDUAL MEMORIES OVERLAP?)
                if memory_bank_id > 0:
                    print("Accuracy on previous memory banks:")
                for k in range(par['n_memory_banks']):
                    if k <= memory_bank_id:
                        par['memory_bank_id'] = k
                        update_dependencies()
                        trial_info = stim.generate_trial(set_rule = None, memory_bank_id = k)
                        h, y, syn_x, syn_u = sess.run([model.h, model.y_output, model.syn_x, model.syn_u],
                            {x: trial_info['neural_input'], t: trial_info['desired_output'], m: trial_info['train_mask']})

                        accuracies = analysis.get_perf_sr(trial_info['desired_output'], y, trial_info['train_mask'])
                        print(f"\tbank {k}: {accuracies}")
                        memory_bank_performance[k].append(accuracies)
                    else:
                        memory_bank_performance[k].append(0)

            i += 1

        # Save out activities
        trial_info = stim.generate_trial(set_rule = None, memory_bank_id = memory_bank_id)
        h, y, syn_x, syn_u = sess.run([model.h, model.y_output, model.syn_x, model.syn_u],
            {x: trial_info['neural_input'], t: trial_info['desired_output'], m: trial_info['train_mask']})

        # Save model and results
        weights = sess.run(model.var_dict)
        save_results(model_performance, weights, save_fn="sequence_reproduction_proof_of_concept.pkl")


################################################################################

def save_results(model_performance, weights, save_fn = None):

    results = {'weights': weights, 'parameters': par}
    for k,v in model_performance.items():
        results[k] = v
    if save_fn is None:
        fn = par['save_dir'] + par['save_fn']
    else:
        fn = par['save_dir'] + save_fn
    pickle.dump(results, open(fn, 'wb'))
    print('Model results saved in ',fn)


def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, iteration):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['weight_loss'].append(weight_loss)
    model_performance['iteration'].append(iteration)

    return model_performance


def print_results(iter_num, perf_loss, spike_loss, weight_loss, h, accuracy):

    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy ' + '{:0.4f}'.format(accuracy) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Mean activity {:0.4f}'.format(np.mean(h)))

def print_important_params():

    important_params = ['num_iterations', 'learning_rate', 'noise_rnn_sd', 'noise_in_sd','spike_cost',\
        'spike_regularization', 'weight_cost','test_cost_multiplier', 'trial_type','balance_EI', 'dt',\
        'delay_time', 'connection_prob','synapse_config','tau_slow','tau_fast']
    for k in important_params:
        print(k, ': ', par[k])
