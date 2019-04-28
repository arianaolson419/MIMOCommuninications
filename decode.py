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
tx_combined = tx_info['combined']

use_saved_signal = False

if not use_saved_signal:
# load received data
    received_data = np.fromfile("Data/MIMOReceive.dat", dtype=np.float32)
    signal_time_rx = received_data[::2] + received_data[1::2]*1j

    plt.plot(signal_time_rx.real)
    plt.show()

# # TODO: Find variables needed to get start of the data chunks using the LTS.
# # functions used: receivers.detect_start_lts
    signal_len = tx_combined.shape[-1]
    lag, signal_time_rx = receivers.detect_start_lts(signal_time_rx, header[0], signal_len)

    plt.plot(signal_time_rx.real)
    plt.show()

    np.savez('isolated_signal.npz', signal_rx=signal_time_rx)

else:
    signal_time_rx = np.load('isolated_signal.npz')[signal_rx]

rx_header_1 = signal_time_rx[:header.shape[-1]]
rx_header_2 = signal_time_rx[header.shape[-1]+mimo.ZERO_SAMPLES:header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1]]

# # # correct for timing differences using timing_offset/f_delta variable
# timing_offset = receivers.estimate_f_delta(lts, num_samples)
# signal_time_timing_corrected = receivers.correct_freq_offset(signal_time_rx, timing_offset)

# # estimate the channel (make sure it is corrected in time first)
# # TODO: tweak channel_estimates function to comply with Alamouti code?
H = receivers.estimate_channel_alamouti(rx_header_1, rx_header_2, header[0], header[1])

print("H:\n", H)

# # received vectors: y_1 = h_1(x_1) + h_2(x_2) + n_1 for 1st block
# # y_2 = h_1(-x*_2) + h_2 * (x*_1) + n_2 2nd block
# # h_1 and h_0 depends on alpha_coefficient * e^(j * theta)

# # based on estimates, H would be [[h_1, h_2], [h*_2, -h1*_1]]
# # calculate s_0 and s_1

rx_data = signal_time_rx[header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1] + mimo.ZERO_SAMPLES: header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1] + mimo.ZERO_SAMPLES + data.shape[-1] - 1]
print("rx_data.shape: ", rx_data.shape)


recovered_signal = receivers.recover_signals_alamouti(rx_data, H)
print(recovered_signal)

plt.plot(np.sign(recovered_signal))
plt.show()

# # Calculate Bit Error Rate (use signal_util.calculate_error_rate() function)
# signal_util.calculate_error_rate()
