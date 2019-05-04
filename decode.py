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
symbols = tx_info['data_bits']


use_saved_signal = False

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
f_delta_header_2, y_scale_header_2 = receivers.find_f_delta(rx_header_2)

print("f_delta from header 2: ", f_delta_header_2)

# plotting constellation before timing correction
plt.plot(signal_time_rx.real[10::20], signal_time_rx.imag[10::20], ".")
plt.title("Received Signal Constellation Before Timing Offset")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()

signal_time_rx = receivers.correct_timing_offset(f_delta_header_2, y_scale_header_2, signal_time_rx)

plt.plot(signal_time_rx)
plt.title("Received Signal After Timing Correction")
plt.show()



# plotting constellation after timing correction
plt.plot(signal_time_rx.real[:header.shape[-1]], signal_time_rx.imag[:header.shape[-1]], ".")
plt.title("Received Header 1 Constellation After Timing Offset")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()

plt.plot(signal_time_rx.real[header.shape[-1]+mimo.ZERO_SAMPLES:header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1]], signal_time_rx.imag[header.shape[-1]+mimo.ZERO_SAMPLES:header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1]], ".")
plt.title("Received Header 2 Constellation After Timing Offset")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()


rx_header_1 = signal_time_rx[:header.shape[-1]]
rx_header_2 = signal_time_rx[header.shape[-1]+mimo.ZERO_SAMPLES:header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1]]



# estimate the channel
H = receivers.estimate_channel_alamouti(rx_header_1, rx_header_2, header[0], header[1])

print("H:\n", H)

rx_data = signal_time_rx[header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1] + mimo.ZERO_SAMPLES: header.shape[-1] + mimo.ZERO_SAMPLES + header.shape[-1] + mimo.ZERO_SAMPLES + data.shape[-1]]

rx_data_downsampled = rx_data[10::20]


recovered_signal = receivers.recover_signals_alamouti(rx_data_downsampled, H) 


plt.plot(recovered_signal.real, recovered_signal.imag, ".")
plt.title("Recovered Signal Constellation")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()




recovered_signal_symbols = receivers.turn_data_to_bits(recovered_signal)

# plt.plot(recovered_signal_bits == tx_qpsk_bits)
# plt.show()
print("error rate (no rotation): ", receivers.calculate_error_rate(symbols, recovered_signal_symbols))


# for n in range(8):
#     rotated_recovered_signal_symbols = receivers.turn_data_to_bits(recovered_signal * np.exp((np.pi/4) * n * 1j))
#     print("error rate: ", receivers.calculate_error_rate(symbols, rotated_recovered_signal_symbols))