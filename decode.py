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

use_saved_signal = True

if not use_saved_signal:
# load received data
    received_data = np.fromfile("Data/MIMOReceive.dat", dtype=np.float32)
    signal_time_rx = received_data[::2] + received_data[1::2]*1j

    plt.plot(signal_time_rx.real)
    plt.show()

    signal_len = tx_combined.shape[-1]
    lag, signal_time_rx = receivers.detect_start_lts(signal_time_rx, header[0], signal_len)

    plt.plot(signal_time_rx.real)
    plt.show()

    np.savez('isolated_signal.npz', signal_rx=signal_time_rx)

else:
    signal_time_rx = np.load('isolated_signal.npz')["signal_rx"]

rx_header_1 = signal_time_rx[:header.shape[-1]]
rx_header_2 = signal_time_rx[header.shape[-1]+mimo.ZERO_SAMPLES:header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1]]

# # # correct for timing differences using timing_offset/f_delta variable
# timing_offset = receivers.estimate_f_delta(lts, num_samples)
# signal_time_timing_corrected = receivers.correct_freq_offset(signal_time_rx, timing_offset)

# # estimate the channel (make sure it is corrected in time first)
H = receivers.estimate_channel_alamouti(rx_header_1, rx_header_2, header[0], header[1])

print("H:\n", H)



rx_data = signal_time_rx[header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1] + mimo.ZERO_SAMPLES: header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1] + mimo.ZERO_SAMPLES + data.shape[-1]]
print("rx_data.shape: ", rx_data.shape)

rx_data_downsampled = rx_data[10::20]

recovered_signal = receivers.recover_signals_alamouti(rx_data_downsampled, H)


print("rx-data shape", rx_data.shape)
print("recovered_signal shape", recovered_signal.shape)

# # Calculate Bit Error Rate
tx_qpsk = np.zeros(data.shape[-1], dtype=np.complex128)
tx_qpsk[::2] = data[0, ::2]
tx_qpsk[1::2] = data[1, ::2]



recovered_signal_bits = receivers.turn_data_to_bits(recovered_signal)

tx_qpsk_bits          = receivers.turn_data_to_bits(tx_qpsk[10::20])

print("error rate: ", receivers.calculate_error_rate(recovered_signal_bits, tx_qpsk_bits))

plt.plot(recovered_signal_bits)
plt.show()

plt.plot(tx_qpsk_bits)
plt.show()

plt.plot(recovered_signal_bits == tx_qpsk_bits)
plt.show()
