import numpy as np
import matplotlib.pyplot as plt
from parameters_sr import *
from sympy.utilities.iterables import multiset_permutations
import itertools
import scipy.misc

class SeedGenerator:

    def __init__(self, pattern_length, seed_length, n_symbols, n_hidden, n_output, time_per, allow_repeated_symbols = False, seed = 42):

        # Shape params
        self.total_pattern_length = pattern_length
        self.pattern_length       = pattern_length - seed_length
        self.n_symbols            = n_symbols
        self.n_hidden             = n_hidden
        self.n_output             = n_output
        self.time_per             = time_per

        # Static params
        self.symbol_set           = np.arange(1, n_symbols + 1)
        self.repeated_symbols     = allow_repeated_symbols

        # Set random seed
        np.random.seed(seed)

    def form_origins(self, patterns):
        """
        Form origin for each pattern.
        """
        k = self.n_hidden // self.n_symbols
        origins = np.zeros((patterns.shape[0], self.n_hidden))
        for i, pattern in enumerate(patterns):
            for j, symbol in enumerate(self.symbol_set):
                ind = np.where(pattern == symbol)[0]
                if len(ind) > 0:
                    origins[i, j*k:(j+1)*k] = ind / self.n_symbols
        return origins

    def make_batch(self, batch_size=1024):

        """
        Generate spray of noise-to-symbol sequence mappings.

        """

        # Generate pattern set
        reproduction_shape = (batch_size, self.pattern_length)

        # Construct remainder of pattern for every seed
        if self.repeated_symbols:
            patterns = np.random.choice(self.symbol_set,
                                        size=reproduction_shape)
        else:
            patterns = np.array([np.random.choice(self.symbol_set,
                                            size=self.pattern_length,
                                            replace=False) for i in range(batch_size)])

        # Come up with associated origin vectors (inputs)
        inputs = self.form_origins(patterns)

        # Form targets
        targets = np.zeros((self.pattern_length*self.time_per, batch_size, self.n_output))
        for i, pattern in enumerate(patterns):
            one_hot_outputs_repr = np.zeros((len(pattern), par['n_output']))
            one_hot_outputs_repr[np.arange(len(pattern)), pattern] = 1.
            repr_outputs = np.repeat(one_hot_outputs_repr, self.time_per, 0)
            targets[:, i, :] = repr_outputs

        # Form mask
        mask = np.ones((batch_size, self.n_output))

        return inputs, targets, mask
