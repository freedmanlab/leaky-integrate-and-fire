import numpy as np
import tensorflow as tf
import os
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'save_fn'               : 'model_results.pkl',

    # Network configuration
    'synapse_config'        : 'full', # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,    # excitatory/inhibitory ratio, set to 1 so that units are neither exc or inh
    'balance_EI'            : True,
    'connection_prob'       : 1.0,

    ####################################
    # Toppings
    ####################################
    'all_toppings'          : False, # If true, turns on all constraints to default values

    # Multi-module
    'multi_module'          : False,
    'input_bottleneck'      : None, # None, or float in [0, 1.) [default: 0.1]
    'input_ei_balance'      : None, # None, or float in [0, 1.) [default: 0.5]
    'output_bottleneck'     : None, # None, or float in [0, 1.) [default: 0.1]
    'output_ei_balance'     : None, # None, or float in [0, 1.) [default: 0.5]

    # Excitability 
    'excitability'          : False,

    # High-throughput
    'high_throughput'       : True,

    # General development tag
    'in_development'        : False,

    ####################################

    # Network shape
    'num_motion_tuned'      : 24,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 400,
    'n_output'              : 3,
    'n_networks'            : 1,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 1e-4,
    'membrane_time_constant': 100,

    # Input and noise
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.5,

    # Tuning function data
    'num_motion_dirs'       : 6,
    'tuning_height'         : 4,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Loss parameters
    'spike_regularization'  : 'L2', # 'L1' or 'L2'
    'spike_cost'            : 2e-2,
    'weight_cost'           : 0.,
    'clip_max_grad_val'     : 0.1,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 512,
    'num_iterations'        : 2000,
    'iters_between_outputs' : 100,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 0,
    'fix_time'              : 500,
    'sample_time'           : 500,
    'delay_time'            : 1000,
    'test_time'             : 500,
    'variable_delay_max'    : 300,
    'mask_duration'         : 40,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task
    'test_cost_multiplier'  : 1.,
    'rule_cue_multiplier'   : 1.,
    'var_delay'             : False,
}



def update_parameters(updates):
    """ Takes a list of strings and values for updating parameters in the parameter dictionary
        Example: updates = [(key, val), (key, val)] """

    #print('Updating parameters...')
    if 'all_toppings' in updates.keys():
        sprinkle_toppings(updates['all_toppings'])
    for key, val in updates.items():
        par[key] = val
    check_toppings()

    update_trial_params()
    update_dependencies()

def sprinkle_toppings(update):
    if update:
        par['multi_module']      = True
        par['excitability']      = True
        par['high_throughput']   = True

        par['input_bottleneck']  = 0.1
        par['output_bottleneck'] = 0.1
        par['input_ei_balance']  = 0.5
        par['output_ei_balance'] = 1.0
        
def check_toppings():

    # First: if multi-module is on, bottlenecks must be, too
    if par['multi_module']:
        if par['input_bottleneck'] is None:
            par['input_bottleneck'] = 0.1
        if par['output_bottleneck'] is None:
            par['output_bottleneck'] = 0.1

    # Next: make sure EI balance set for bottlenecks, if needed
    if par['input_bottleneck'] is not None:
        if par['input_ei_balance'] is None:
            par['input_ei_balance'] = 0.5
    if par['output_bottleneck'] is not None:
        if par['output_ei_balance'] is None:
            par['output_ei_balance'] = 1.0


def update_trial_params():
    """ Update all the trial parameters given trial_type """
    par['num_rules'] = 1
    par['num_receptive_fields'] = 1
    #par['num_rule_tuned'] = 0
    par['ABBA_delay' ] = 0
    par['rule_onset_time'] = [par['dead_time']]
    par['rule_offset_time'] = [par['dead_time']]

    if par['trial_type'] == 'DMS' or par['trial_type'] == 'DMC':
        par['rotation_match'] = 0
        par['n_output'] = 3

    elif par['trial_type'] == 'SR':

        # Pattern parameters
        par['n_patterns']             = 50
        par['n_memory_banks']         = 100
        par['symbol_set_length']      = 25
        par['symbol_time']            = 200
        par['seed_length']            = 3
        par['pattern_length']         = 25
        par['allow_repeated_symbols'] = False
        par['show_patterns']          = False
        par['bank_overlap_pairwise']  = 0.0 # NOT YET FLESHED OUT
        par['bank_overlap_total']     = 0.0 # NOT YET FLESHED OUT  
        par['memory_banks']           = dict()
        par['memory_bank_id']         = 0

        par['n_hidden'] = par['symbol_set_length'] * 10

        # Timing parameters
        par['delay_time'] = par['symbol_time']
        par['dead_time']  = 0

        # Network parameters
        par['n_input']          = par['symbol_set_length'] + 2
        par['n_output']         = par['symbol_set_length'] + 1
        par['num_fix_tuned']    = 1
        par['num_motion_tuned'] = par['n_input'] - par['num_fix_tuned']

        # Excitability parameters: long timescale
        par['t_since_update']      = 0 
        par['excitability_period'] = 0 
        par['dyn_rng_long']        = [0.25, 2]
        par['long_phases']         = np.zeros((par['n_hidden']))

        # Excitability parameters: short timescale
        par['dyn_rng_short'] = [0.5, 1.5]

        # Granularity
        par['granularity'] = 2

        # Modularity parameters
        c = 0.05
        par['Wf0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))
        par['Wi0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))
        par['Wo0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))
        par['Wc0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))

        par['Uf0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))
        par['Ui0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))
        par['Uo0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))
        par['Uc0'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_hidden']]))

        par['bf0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)
        par['bi0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)
        par['bo0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)
        par['bc0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)

        # FOR FUTURE REFERENCE: EXCITABILITY GROUPS SET BELOW

    elif par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif par['trial_type'] == 'DMRS90ccw':
        par['rotation_match'] = -90

    elif  par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

    elif par['trial_type'] == 'dualDMS':
        par['catch_trial_pct'] = 0
        par['num_receptive_fields'] = 2
        par['num_rules'] = 2
        par['probe_trial_pct'] = 0
        par['probe_time'] = 10
        par['num_rule_tuned'] = 6
        par['sample_time'] = 500
        par['test_time'] = 500
        par['delay_time'] = 1000
        par['analyze_rule'] = True
        par['num_motion_tuned'] = 24*2
        par['rule_onset_time'] = []
        par['rule_offset_time'] = []
        par['rule_onset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + par['delay_time']/2)
        par['rule_offset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + par['delay_time'] + par['test_time'])
        par['rule_onset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + 3*par['delay_time']/2 + par['test_time'])
        par['rule_offset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + 2*par['delay_time'] + 2*par['test_time'])


    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['sample_time'] = 400
        par['ABBA_delay'] = 400
        par['delay_time'] = 6*par['ABBA_delay']
        par['repeat_pct'] = 0
        par['analyze_test'] = False
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif 'DMS+DMRS' in par['trial_type'] and par['trial_type'] != 'DMS+DMRS+DMC':

        par['num_rules'] = 2
        par['num_rule_tuned'] = 6
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 500]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 750]
        elif par['trial_type'] == 'DMS+DMRS_full_cue':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time']\
                +par['delay_time']+par['test_time']]
        else:
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']]

    elif par['trial_type'] == 'DMS+DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rotation_match'] = [0, 0]
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]
        par['n_output'] = 3

    elif par['trial_type'] == 'DMS+DMRS+DMC':
        par['num_rules'] = 3
        par['num_rule_tuned'] = 18
        par['rotation_match'] = [0, 90, 0]
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]
        if par['in_development']:
            par['n_output'] = 2
        else:
            par['n_output'] = 3

    elif par['trial_type'] == 'location_DMS':
        par['num_receptive_fields'] = 3
        par['rotation_match'] = 0
        par['num_motion_tuned'] = 24*3

    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()

    if par['trial_type'] == 'dualDMS':
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    elif par['trial_type'] == 'SR':
        if par['show_patterns']:
            par['trial_length'] = (par['dead_time'] + 2 * par['symbol_time']*par['pattern_length'] \
                                        + par['delay_time'])
        else:
            par['trial_length'] = (par['dead_time'] + par['symbol_time']*par['pattern_length'])
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


def update_dependencies():
    """ Updates all parameter dependencies """
    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms

    par['dead_time_rng'] = range(par['dead_time']//par['dt'])
    par['sample_time_rng'] = range((par['dead_time']+par['fix_time'])//par['dt'], (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'])
    par['rule_time_rng'] = [range(int(par['rule_onset_time'][n]/par['dt']), int(par['rule_offset_time'][n]/par['dt'])) for n in range(len(par['rule_onset_time']))]

    # If exc_inh_prop is < 1, then neurons can be either excitatory or
    # inihibitory; if exc_inh_prop = 1, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['ind_inh'] = np.where(par['EI_list']==-1)[0]

    # EI matrix
    if not par['high_throughput']:
        par['EI_matrix'] = np.diag(par['EI_list'])
    else:
        par['EI_matrix'] = np.tile(np.diag(par['EI_list'])[np.newaxis, :, :], (par['n_networks'], 1, 1))

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    # initial neural activity
    if not par['high_throughput']:
        par['h0'] = 0.1*np.ones((1, par['n_hidden']), dtype=np.float32)
    else:
        par['h0'] = 0.1*np.ones((par['n_networks'], 1, par['n_hidden']), dtype=np.float32)


    # initial input weights
    if not par['high_throughput']:
        par['w_in0'] = initialize([par['n_input'], par['n_hidden']], par['connection_prob']/par['num_receptive_fields'], shape=0.2, scale=1.)
    else:
        par['w_in0'] = initialize([par['n_networks'], par['n_input'], par['n_hidden']], par['connection_prob']/par['num_receptive_fields'], shape=0.2, scale=1.)
    if par['input_bottleneck'] is not None: # NOTE: if input bottleneck is enabled, make it high-throughput
        # Units to shut off 
        n_receivers = int(par['input_bottleneck'] * par['n_hidden'])
        first_receiver = par['num_exc_units'] - int(par['input_ei_balance'] * n_receivers)
        off_inds = list(range(0, first_receiver))
        off_inds.extend(list(range(first_receiver + n_receivers, par['n_hidden'])))
        #off_inds = list(range(0, par['num_exc_units'] - 20))
        #off_inds.extend(list(range(par['num_exc_units'] + 20, par['n_hidden']))) 
        par['w_in0'][:,:,off_inds] = 0

        # New initialization
        w_in_sub = initialize([par['n_networks'], par['n_input'], n_receivers], par['connection_prob']/par['num_receptive_fields'], shape=0.2, scale=1.)
        par['w_in0'][:,:,first_receiver:first_receiver + n_receivers] = w_in_sub
        #par['w_in0'][:,:,par['num_exc_units']-20:par['num_exc_units']+20] = w_in_sub

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        ## OLD:
        if not par['high_throughput']:
            par['w_rnn0'] = initialize([par['n_hidden'], par['n_hidden']], par['connection_prob'])
        else:
            par['w_rnn0'] = initialize([par['n_networks'], par['n_hidden'], par['n_hidden']], par['connection_prob'])
        if par['balance_EI']:
            # increase the weights to and from inh units to balance excitation and inhibition
            if not par['high_throughput']:
                par['w_rnn0'][:, par['ind_inh']] = initialize([par['n_hidden'], par['num_inh_units']], par['connection_prob'], shape=0.2, scale=1.)
                par['w_rnn0'][par['ind_inh'], :] = initialize([par['num_inh_units'], par['n_hidden']], par['connection_prob'], shape=0.2, scale=1.)
            else:
                par['w_rnn0'][:, :, par['ind_inh']] = initialize([par['n_networks'], par['n_hidden'], par['num_inh_units']], par['connection_prob'], shape=0.2, scale=1.)
                par['w_rnn0'][:, par['ind_inh'], :] = initialize([par['n_networks'], par['num_inh_units'], par['n_hidden']], par['connection_prob'], shape=0.2, scale=1.)

    else:
        if not par['high_throughput']:
            par['w_rnn0'] = 0.54*np.eye(par['n_hidden'])
        else:
            par['w_rnn0'] = 0.54*np.eye(par['n_hidden'])
            par['w_rnn0'] = np.tile(par['w_rnn0'][np.newaxis,:,:], (par['n_networks'], 1, 1))


    # handle modularity
    if par['multi_module']:

        # 0. Setup
        n_receivers = int(par['input_bottleneck'] * par['n_hidden'])
        first_receiver = par['num_exc_units'] - int(par['input_ei_balance'] * n_receivers)
        last_receiver = first_receiver + n_receivers
        n_projectors = int(par['output_bottleneck'] * par['n_hidden'])
        first_projector = 0
        last_projector = first_projector + n_projectors

        # set up some useful indexings
        module_1_direct = list(range(first_receiver, last_receiver))
        module_1_indirect = [list(range(par['num_exc_units'] // 2, first_receiver))]
        module_1_indirect.append(list(range(last_receiver, par['num_exc_units'] + par['num_inh_units'] // 2)))
        module_2_direct = list(range(first_projector, last_projector))
        module_2_indirect = [list(range(last_projector, par['num_exc_units'] // 2))]
        module_2_indirect.append(list(range(module_1_indirect[1][-1] + 1, par['n_hidden'])))
        
        module_1_inh = list(range(par['num_exc_units'], par['num_exc_units'] + par['num_inh_units'] // 2))
        module_2_inh = list(range(par['num_exc_units'] + par['num_inh_units'] // 2, par['n_hidden']))
        '''print(module_1_inh)
        print(module_2_inh)
        print(0/0)'''

        # 1. CHECKED: Ensure units receiving input/output don't project directly to one another
        par['w_rnn0'][:, first_receiver:last_receiver, first_projector:last_projector] = 0
        par['w_rnn0'][:, first_projector:last_projector, first_receiver:last_receiver] = 0

        # 1a. Ensure units that are directly receiving input/constructing output don't receive inputs from
        # out-of-module units
        for i in range(2):
            par['w_rnn0'][:, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1, module_2_direct[0]:module_2_direct[-1] + 1] = 0
            par['w_rnn0'][:, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1, module_1_direct[0]:module_1_direct[-1] + 1] = 0
        ## OLD
        #par['w_rnn0'][:, par['num_exc_units'] - 20:par['num_exc_units']+20, 0:40] = 0
        #par['w_rnn0'][:, 0:40, par['num_exc_units'] - 20:par['num_exc_units']+20] = 0

        # 2. Enforce 2 modules: inhibitory neurons (not direct receivers) don't make projections out-of-module
        #par['w_rnn0'][:, par['num_exc_units']:last_receiver, 0:par['n_hidden'] // 2] = 0 # Module 1
        #par['w_rnn0'][:, par['num_exc_units']:last_receiver, module_1_indirect[1][-1]:] = 0
        #par['w_rnn0'][:, module_1_indirect[1][-1]:, par['n_hidden'] // 2:module_1_indirect[-1]] = 0 # Module 2
        par['w_rnn0'][:, module_1_inh[0]:module_1_inh[-1] + 1, module_2_direct[0]:module_2_direct[-1] + 1] = 0
        par['w_rnn0'][:, module_2_inh[0]:module_2_inh[-1] + 1, module_1_direct[0]:module_1_direct[-1] + 1] = 0
        for i in range(2):
            par['w_rnn0'][:, module_1_inh[0]:module_1_inh[-1] + 1, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1] = 0
            par['w_rnn0'][:, module_2_inh[0]:module_2_inh[-1] + 1, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1] = 0

        ## OLD
        #par['w_rnn0'][:,320:360, 0:160] = 0 # Module 1
        #par['w_rnn0'][:,320:360, 360:400] = 0
        #par['w_rnn0'][:,360:400, 160:360] = 0 # Module 2

        # 3. Direct input/output neurons don't project out of module
        for i in range(2):
            par['w_rnn0'][:, module_1_direct[0]:module_1_direct[-1] + 1, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1] = 0
            par['w_rnn0'][:, module_2_direct[0]:module_2_direct[-1] + 1, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1] = 0

        ## OLD
        #par['w_rnn0'][:, 0:40, 160:360] = 0
        #par['w_rnn0'][:, 300:340, 0:160] = 0
        #par['w_rnn0'][:, 300:340, 360:400] = 0

        # 4. Re-initialize projections inside each module (1-4)
        '''m1_dir = initialize([par['n_networks'], len(module_1_direct), len(module_1_direct)], par['connection_prob'])
        m1_indir = initialize([par['n_networks'], sum([len(i) for i in module_1_indirect]), sum([len(i) for i in module_1_indirect])], par['connection_prob'])
        m2_indir = initialize([par['n_networks'], sum([len(i) for i in module_2_indirect]), sum([len(i) for i in module_2_indirect])], par['connection_prob'])
        m2_dir = initialize([par['n_networks'], len(module_2_direct), len(module_2_direct)], par['connection_prob'])
        
        # 4a. Direct-to-direct
        par['w_rnn0'][:, module_1_direct[0]:module_1_direct[-1] + 1, module_1_direct[0]:module_1_direct[-1] + 1] = m1_dir
        par['w_rnn0'][:, module_2_direct[0]:module_2_direct[-1] + 1, module_2_direct[0]:module_2_direct[-1] + 1] = m2_dir

        # 4b. Indirect-to-indirect
        start_inds = np.array([[0, 0], [len(module_1_indirect[0]), len(module_2_indirect[0])]])
        for i in range(2):
            print(start_inds[i,0] + len(module_1_indirect[i]))
            par['w_rnn0'][:, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1] = m1_indir[:, start_inds[i,0]:start_inds[i,0] + len(module_1_indirect[i]), start_inds[i,0]:start_inds[i,0] + len(module_1_indirect[i])]
            par['w_rnn0'][:, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1] = m2_indir[:, start_inds[i,1]:start_inds[i,1] + len(module_2_indirect[i]), start_inds[i,1]:start_inds[i,1] + len(module_2_indirect[i])]

            j = (i + 1) % 2
            par['w_rnn0'][:, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1, module_1_indirect[j][0]:module_1_indirect[j][-1] + 1] = m1_indir[:, start_inds[i,0]:start_inds[i,0] + len(module_1_indirect[i]), start_inds[j,0]:start_inds[j,0] + len(module_1_indirect[j])]
            par['w_rnn0'][:, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1, module_2_indirect[j][0]:module_2_indirect[j][-1] + 1] = m2_indir[:, start_inds[i,1]:start_inds[i,1] + len(module_2_indirect[i]), start_inds[j,1]:start_inds[j,1] + len(module_2_indirect[j])]'''

        #par['w_rnn0'][:, module_1_indirect[0][0]:module_1_indirect[0][-1], module_1_indirect[i][0]:module_1_indirect[i][-1]] = m1_indir[:, 0:len(module_1_indirect[0]), 0:len(module_1_indirect[0])]
        #par['w_rnn0'][:, module_2_indirect[0][0]:module_2_indirect[0][-1], module_2_indirect[i][0]:module_2_indirect[i][-1]] = m2_indir[:, 0:len(module_2_indirect[0]), 0:len(module_2_indirect[0])]
        
        #par['w_rnn0'][:, module_1_indirect[i][0]:module_1_indirect[i][-1], module_1_indirect[i][0]:module_1_indirect[i][-1]] = m1_indir[:, 0:len(module_1_indirect[0]), 0:len(module_1_indirect[0])]
        #par['w_rnn0'][:, module_2_indirect[i][0]:module_2_indirect[i][-1], module_2_indirect[i][0]:module_2_indirect[i][-1]] = m2_indir[:, 0:len(module_2_indirect[0]), 0:len(module_2_indirect[0])]


        # 5. Re-initialize projections between each module (1a/b, 2a/b, 3a/b)
        #m1_dir_forward = initialize([par['n_networks'], len(module_1_direct), sum([len(i) for i in module_1_indirect])], par['connection_prob'])
        #m1_dir_backward = initialize([par['n_networks'], sum([len(i) for i in module_1_indirect]), len(module_1_direct)], par['connection_prob'])
        m1_indir_forward = initialize([par['n_networks'], sum([len(i) for i in module_1_indirect]), sum([len(i) for i in module_2_indirect])], par['connection_prob_ff'])
        m1_indir_backward = initialize([par['n_networks'], sum([len(i) for i in module_2_indirect]), sum([len(i) for i in module_1_indirect])], par['connection_prob_fb'])
        #m2_indir_forward = initialize([par['n_networks'], sum([len(i) for i in module_2_indirect]), len(module_2_direct)], par['connection_prob'])
        #m2_indir_backward = initialize([par['n_networks'], len(module_2_direct), sum([len(i) for i in module_2_indirect])], par['connection_prob'])
        start_inds = np.array([[0, 0], [len(module_1_indirect[0]), len(module_2_indirect[0])]])
        for i in range(2):
            #print(start_inds[i,0] + len(module_1_indirect[i]))
            par['w_rnn0'][:, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1] = m1_indir_forward[:, start_inds[i,0]:start_inds[i,0] + len(module_1_indirect[i]), start_inds[i,1]:start_inds[i,1] + len(module_2_indirect[i])]
            par['w_rnn0'][:, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1] = m1_indir_backward[:, start_inds[i,1]:start_inds[i,1] + len(module_2_indirect[i]), start_inds[i,0]:start_inds[i,0] + len(module_1_indirect[i])]

            j = (i + 1) % 2
            par['w_rnn0'][:, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1, module_2_indirect[j][0]:module_2_indirect[j][-1] + 1] = m1_indir_forward[:, start_inds[i,0]:start_inds[i,0] + len(module_1_indirect[i]), start_inds[j,1]:start_inds[j,1] + len(module_2_indirect[j])]
            par['w_rnn0'][:, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1, module_1_indirect[j][0]:module_1_indirect[j][-1] + 1] = m1_indir_backward[:, start_inds[i,1]:start_inds[i,1] + len(module_2_indirect[i]), start_inds[j,0]:start_inds[j,0] + len(module_1_indirect[j])]

            # 1b: Module 1 Indirect -> Module 1 Direct

            # 2a: Module 1 Indirect -> Module 2 Indirect

            # 2b: Module 2 Indirect -> Module 1 Indirect

            # 3a: Module 2 Indirect -> Module 2 Direct

            # 3b: Module 2 Direct -> Module 2 Indirect
        '''par['w_rnn0'][:, module_1_direct[0]:module_1_direct[-1] + 1, module_1_indirect[0]:module_1_indirect[-1] + 1] = m1_dir_forward
        par['w_rnn0'][:, module_1_indirect[0]:module_1_indirect[-1] + 1, module_1_direct[0]:module_1_direct[-1] + 1] = m1_dir_backward
        par['w_rnn0'][:, module_1_indirect[0]:module_1_indirect[-1] + 1, module_2_indirect[0]:module_2_indirect[-1] + 1] = m1_indir_forward
        par['w_rnn0'][:, module_2_indirect[0]:module_2_indirect[-1] + 1, module_1_indirect[0]:module_1_indirect[-1] + 1] = m1_indir_backward
        par['w_rnn0'][:, module_2_indirect[0]:module_2_indirect[-1] + 1, module_2_direct[0]:module_2_direct[-1] + 1] = m2_indir_forward
        par['w_rnn0'][:, module_2_direct[0]:module_2_direct[-1] + 1, module_2_indirect[0]:module_2_indirect[-1] + 1] = m2_indir_backward'''

        # 5. Write out direct vs. indirect, module by module, to facilitate later analysis

        par['w_rnn0'][:, module_1_inh[0]:module_1_inh[-1] + 1, module_2_direct[0]:module_2_direct[-1] + 1] = 0
        par['w_rnn0'][:, module_2_inh[0]:module_2_inh[-1] + 1, module_1_direct[0]:module_1_direct[-1] + 1] = 0
        for i in range(2):
            par['w_rnn0'][:, module_1_inh[0]:module_1_inh[-1] + 1, module_2_indirect[i][0]:module_2_indirect[i][-1] + 1] = 0
            par['w_rnn0'][:, module_2_inh[0]:module_2_inh[-1] + 1, module_1_indirect[i][0]:module_1_indirect[i][-1] + 1] = 0
        par['module_1_direct']   = module_1_direct
        par['module_1_indirect'] = module_1_indirect
        par['module_2_direct']   = module_2_direct
        par['module_2_indirect'] = module_2_indirect


    # initial recurrent biases
    if not par['high_throughput']:
        par['b_rnn0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)
    else:
        par['b_rnn0'] = np.zeros((par['n_networks'], 1, par['n_hidden']), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] is None:
        par['w_rnn0'] = par['w_rnn0']/3.

    # initial output weights and biases
    if not par['high_throughput']:
        par['w_out0'] = initialize([par['n_hidden'], par['n_output']], par['connection_prob'])
        par['b_out0'] = np.zeros((1, par['n_output']), dtype=np.float32)
    else:
        par['w_out0'] = initialize([par['n_networks'], par['n_hidden'], par['n_output']], par['connection_prob'])
        par['b_out0'] = np.zeros((par['n_networks'], 1, par['n_output']), dtype=np.float32)

    if par['output_bottleneck']:

        n_projectors = int(par['output_bottleneck'] * par['n_hidden'])
        first_projector = 0
        last_projector = first_projector + n_projectors

        w_out_sub = initialize([par['n_networks'], n_projectors, par['n_output']], par['connection_prob'])
        par['w_out0'][:, last_projector:par['n_hidden'], :] = 0
        par['w_out0'][:, first_projector:last_projector, :] = w_out_sub

        ## OLD
        #par['w_out0'][:,40:par['n_hidden'], :] = 0
        #par['w_out0'][:,0:40,:] = w_out_sub


    # for EI networks, masks will prevent self-connections, and inh to output connections
    par['w_rnn_mask'] = np.ones_like(par['w_rnn0'])
    par['w_out_mask'] = np.ones_like(par['w_out0'])
    par['w_in_mask'] = np.ones_like(par['w_in0'])
    if par['EI']:
        if not par['high_throughput']:
            par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
            par['w_out_mask'][par['ind_inh'], :] = 0
        else:
            par['w_rnn_mask'] = np.ones((par['n_networks'], par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
            par['w_out_mask'][:, par['ind_inh'], :] = 0

    par['w_rnn0'] *= par['w_rnn_mask']
    par['w_out0'] *= par['w_out_mask']

    # for the location_DMS task, inputs from the 3 receptive fields project onto non-overlapping
    # units in the RNN. This tries to replicates what likely happesn in areas MST, which are retinotopic
    if par['trial_type'] == 'location_DMS':
        par['w_in_mask'] *= 0
        target_ind = [range(0, par['n_hidden'],3), range(1, par['n_hidden'],3), range(2, par['n_hidden'],3)]
        for n in range(par['n_input']):
            u = int(n//(par['n_input']/3))
            if not par['high_throughput']:
                par['w_in_mask'][n, target_ind[u]] = 1
            else:
                par['w_in_mask'][:, n, target_ind[u]] = 1
        par['w_in0'] = par['w_in0']*par['w_in_mask']

    synaptic_configurations = {
        'full'              : ['facilitating' if i%2==0 else 'depressing' for i in range(par['n_hidden'])],
        'fac'               : ['facilitating' for i in range(par['n_hidden'])],
        'dep'               : ['depressing' for i in range(par['n_hidden'])],
        'exc_fac'           : ['facilitating' if par['EI_list'][i]==1 else 'static' for i in range(par['n_hidden'])],
        'exc_dep'           : ['depressing' if par['EI_list'][i]==1 else 'static' for i in range(par['n_hidden'])],
        'inh_fac'           : ['facilitating' if par['EI_list'][i]==-1 else 'static' for i in range(par['n_hidden'])],
        'inh_dep'           : ['depressing' if par['EI_list'][i]==-1 else 'static' for i in range(par['n_hidden'])],
        'exc_dep_inh_fac'   : ['depressing' if par['EI_list'][i]==1 else 'facilitating' for i in range(par['n_hidden'])],
        'none'              : [None for i in range(par['n_hidden'])]

    }

    # initialize synaptic values
    if not par['high_throughput']:
        par['alpha_stf'] = np.ones((1, par['n_hidden']), dtype=np.float32)
        par['alpha_std'] = np.ones((1, par['n_hidden']), dtype=np.float32)
        par['U'] = np.ones((1, par['n_hidden']), dtype=np.float32)
        par['syn_x_init'] = np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)
        par['syn_u_init'] = 0.3 * np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)
        par['dynamic_synapse'] = np.zeros((1, par['n_hidden']), dtype=np.float32)
    else:
        par['alpha_stf'] = np.ones((par['n_networks'], 1, par['n_hidden']), dtype=np.float32)
        par['alpha_std'] = np.ones((par['n_networks'], 1, par['n_hidden']), dtype=np.float32)
        par['U'] = np.ones((par['n_networks'], 1, par['n_hidden']), dtype=np.float32)
        par['syn_x_init'] = np.ones((par['n_networks'], par['batch_size'], par['n_hidden']), dtype=np.float32)
        par['syn_u_init'] = 0.3 * np.ones((par['n_networks'], par['batch_size'], par['n_hidden']), dtype=np.float32)
        par['dynamic_synapse'] = np.zeros((par['n_networks'], 1, par['n_hidden']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if synaptic_configurations[par['synapse_config']][i] == 'facilitating':
            if not par['high_throughput']:
                par['alpha_stf'][0,i] = par['dt']/par['tau_slow']
                par['alpha_std'][0,i] = par['dt']/par['tau_fast']
                par['U'][0,i] = 0.15
                par['syn_u_init'][:, i] = par['U'][0,i]
                par['dynamic_synapse'][0,i] = 1
            else:
                par['alpha_stf'][:,0,i] = par['dt']/par['tau_slow']
                par['alpha_std'][:,0,i] = par['dt']/par['tau_fast']
                par['U'][:,0,i] = 0.15
                par['syn_u_init'][:,:, i] = np.tile(par['U'][:,0,i][:,np.newaxis], (1, par['batch_size']))
                par['dynamic_synapse'][:,0,i] = 1

        elif synaptic_configurations[par['synapse_config']][i] == 'depressing':
            if not par['high_throughput']:
                par['alpha_stf'][0,i] = par['dt']/par['tau_fast']
                par['alpha_std'][0,i] = par['dt']/par['tau_slow']
                par['U'][0,i] = 0.45
                par['syn_u_init'][:, i] = par['U'][0,i]
                par['dynamic_synapse'][0,i] = 1
            else:
                par['alpha_stf'][:,0,i] = par['dt']/par['tau_fast']
                par['alpha_std'][:,0,i] = par['dt']/par['tau_slow']
                par['U'][:,0,i] = 0.45
                par['syn_u_init'][:,:, i] = np.tile(par['U'][:,0,i][:,np.newaxis], (1, par['batch_size']))
                par['dynamic_synapse'][:,0,i] = 1

    # update excitability state
    if par['excitability']:

        par['exc_values'] = np.linspace(par['dyn_rng_long'][0], par['dyn_rng_long'][1], par['granularity'])

        # Generate groups: exc/inh separately
        exc_groups = list(np.arange(par['granularity']))*(par['num_exc_units'] // par['granularity'])
        inh_groups = list(np.arange(par['granularity']))*(par['num_inh_units'] // par['granularity'])



        exc_groups = np.tile(np.arange(par['granularity']), par['num_exc_units'] // par['granularity'])
        inh_groups = np.tile(np.arange(par['granularity']), par['num_inh_units'] // par['granularity'])
        par['total_groups'] = np.hstack((exc_groups, inh_groups))
        #print(par['memory_bank_id'])
        #if par['memory_bank_id'] == 1:
        #    print(0/0)
        #print(par['total_groups'])
        
        # Generate states
        par['exc_states'] = np.array([par['exc_values'][(par['memory_bank_id'] + i) % par['granularity']] for i in par['total_groups']])

    # For trial type SR
    par['new_init'] = par['w_rnn0']
    ## For SR task:
    #if par['trial_type'] == 'SR':
    #    par['excitability_states'] = np.zeros((par['n_networks', 'n_hidden']))


def initialize(dims, connection_prob, shape=0.1, scale=1.0 ):
    w = np.random.gamma(shape, scale, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)

    return np.float32(w)


update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
