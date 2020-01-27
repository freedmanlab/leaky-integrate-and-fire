## Author: Matt Rosen (2020); modified from Masse/Grant

import numpy as np
import tensorflow as tf
import pickle
import time
import behavior_trainer
import sys
from parameters_sr import *

class BehaviorModule:

    def __init__(self, gpu_id):

        # Reset tf graph
        tf.reset_default_graph()

        # Train on sequence reproduction task
        self.batch_size = 512

        timesteps = (par['pattern_length'] - par['seed_length']) * (par['symbol_time'] // par['dt'])

        input_data  = tf.placeholder(tf.float32, [self.batch_size, par['n_hidden']], 'seed')
        target_data = tf.placeholder(tf.float32, [timesteps, self.batch_size, par['n_output']], 'target')
        mask        = tf.placeholder(tf.float32, [self.batch_size, par['n_output']])

        with tf.Session(config=tf.ConfigProto()) as sess:

            print(gpu_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

            device = '/cpu:0' if gpu_id is None else '/gpu:3'
            print(device)
            with tf.device(device):
                behavior_generator = self.model(input_data, target_data, mask)

            #print(0/0)
            sess.run(tf.global_variables_initializer())
            t_start = time.time()

            s = behavior_trainer.SeedGenerator(par['pattern_length'], par['seed_length'], \
                par['symbol_set_length'], par['n_hidden'], par['n_output'], par['symbol_time'] // par['dt'])
            losses = []

            # Train LSTM part of the network to produce the sequences
            for i in range(par['n_train_batches_lstm']):

                x, y, m = s.make_batch(self.batch_size) 

                _, loss, pred_y  = sess.run([self.train_op, self.loss, self.y], feed_dict={input_data: x, target_data: y, mask: m})
                losses.append(loss)

                if i % 100 == 0:
                    print(f"Iteration {i}: Loss {np.mean(np.array(losses[-100:]))}\n\tTotal elapsed time: {time.time() - t_start}")

                    # Pull out predicted y, pull out desired y, print

                    print(pred_y.shape, y.shape)



            W = {}
            for var in tf.trainable_variables():
                W[var.op.name] = var.eval()

            fn = par['save_dir'] + "LSTM_weights_SR.pkl"

            pickle.dump(W, open(fn, 'wb'))
            print(f"LSTM weights saved in {fn}.")

    def model(self, input_data, target_data, mask):

        self.input_data  = tf.unstack(input_data, axis=0)
        self.target_data = target_data
        self.mask        = mask

        self.initialize_weights()

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    def initialize_weights(self):
        # Initialize all weights. biases, and initial values

        self.var_dict = {}
        to_initialize = ['Wf', 'Wi', 'Wc', 'Wo', 'Uf', 'Ui', 'Uc', 'Uo', 'bf', 'bi', 'bc', 'bo', 'w_out', 'b_out', 'h']
        for k in to_initialize:
            self.var_dict[k] = tf.Variable(par[k + "0"], k)

    def run_model(self):
        # Main model loop

        self.y = []

        h = self.var_dict['h']#tf.zeros_like(par['h0'])
        c = tf.zeros_like(par['h0'])

        for t in range(self.target_data.shape[0]):

            if t == 0:
                h, c = self.lstm(self.input_data, h, c)
            else:
                h, c = self.lstm(tf.zeros_like(self.input_data), h, c)
            self.y.append(tf.squeeze(h @ tf.nn.relu(self.var_dict['w_out']) + self.var_dict['b_out'] - (1 - self.mask)*1e16))

        # Stack results together 
        self.y = tf.stack(self.y)

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

    def optimize(self):

        # loss
        print(self.y.shape, self.target_data.shape)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y, labels=self.target_data, axis=2))

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.train_op = opt.minimize(self.loss)
        

if __name__ == "__main__":
    update_parameters({'trial_type'     : 'SR', 
                       'excitability'   : True,
                       'n_train_batches_lstm': 5000,
                       'learning_rate': 0.01})
    bm = BehaviorModule(sys.argv[1])

    print("Done")

