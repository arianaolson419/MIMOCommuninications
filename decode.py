"""This script decodes received MIMO data.
"""

import numpy as np
import matplotlib.pyplot as plt

import utils.mimo_util as mimo

# TODO: Update this when saving in encode.py is updated.
tx_info = np.load('tx_info.npz')
header = tx_info['header']
data = tx_info['data']
