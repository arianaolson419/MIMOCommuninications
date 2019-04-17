"""This file contains functions used to imlement MIMO in hardware with space-time coding.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

# Constants used for signal creation and analysis.
SIGNAL_PERIOD = 20 # samples
HEADER_SYMBOLS = 100
DATA_SYMBOLS = 1000

def generate_symbols(num_symbols, num_channels, seed):
    """Generate multiple channels of random complex data symbols

    Args:
        num_symbols (int): The number of symbols per channel.
        num_channels (int): The number of channels of data.
        seed (int): The seed used by the random number generator.

    Returns:
        symbols (complex ndarray): An array of size (num_channels, num_symbols)
        symbols of +/-1 +/-1j.
    """
    pass
