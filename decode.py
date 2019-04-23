"""This script decodes received MIMO data.
"""

import numpy as np
import matplotlib.pyplot as plt
import utils.receivers as receivers
import utils.mimo_util as mimo

# TODO: Update this when saving in encode.py is updated.
tx_info = np.load('tx_info.npz')
header = tx_info['header']
data = tx_info['data']
tx_combined = tx_info['tx_combined']

# lts = tx_info['lts']


# load received data
received_data = np.fromfile("Data/MIMOReceive.dat", dtype=np.float32)
signal_time_rx = received_data[::2] + received_data[1::2]*1j

# # 
# plt.plot(signal_time_rx.real)
# plt.plot(signal_time_rx.imag)
# plt.show()

# # TODO: Find variables needed to get start of the data chunks using the LTS.
# # functions used: receivers.detect_start_lts
signal_time_len = tx_combined.shape[-1]
lag, signal_time_rx = receivers.detect_start_lts(signal_time_rx, header[0], signal_time_len)

# # # correct for timing differences using timing_offset/f_delta variable
# timing_offset = receivers.estimate_f_delta(lts, num_samples)
# signal_time_timing_corrected = receivers.correct_freq_offset(signal_time_rx, timing_offset)

# # estimate the channel (make sure it is corrected in time first)
# # TODO: tweak channel_estimates function to comply with Alamouti code?
# channel_estimates = receivers.estimate_channel_alamouti(rx_sections, tx_headers)

# # received vectors: y_1 = h_1(x_1) + h_2(x_2) + n_1 for 1st block
# # y_2 = h_1(-x*_2) + h_2 * (x*_1) + n_2 2nd block
# # h_1 and h_0 depends on alpha_coefficient * e^(j * theta)

# # based on estimates, H would be [[h_1, h_2], [h*_2, -h1*_1]]
# # calculate s_0 and s_1


# # Decode signal by checking for alpha_coefficients and 


# # Calculate Bit Error Rate (use signal_util.calculate_error_rate() function)
