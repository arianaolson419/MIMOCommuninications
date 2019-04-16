"""This script contains code to generate and encode data to transmit using space-time block coding.
"""

import numpy as np
import matplotlib.pyplot as plt

# TODO: replace dummy data with real data.
header = np.zeros((2, 1000))
data = np.zeros((2, 10000))

# TODO: confirm that this is the data we want to save.
# Save data for decoding and BER calculations.
np.savez('tx_info.npz', header=header, data=data)
