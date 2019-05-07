"""This script decodes received MIMO data.

Authors: Annie Ku, Ariana Olson
"""
import numpy as np
import matplotlib.pyplot as plt
import utils.mimo_util as mimo
import argparse

# Set up commandline arguments.
FLAGS = None

parser = argparse.ArgumentParser()
parser.add_argument('--received_data_path',
                    type=str,
                    default='Data/MIMOReceive.dat',
                    help='The path of the binary file containing received data')
parser.add_argument('--isolated_signal_path',
                    type=str,
                    default='isolated_signal.npz',
                    help="""The path of a .npz file containing the isolated
                    data signal obtained using cross correlation on the full
                    received signal.""")
parser.add_argument('--tx_info_path',
                    type=str,
                    default='tx_info.npz',
                    help="""The path of the transmission information generated
                    by encode.py.""")
parser.add_argument('--figure_directory',
                    type=str,
                    default='figures',
                    help='The directory to save images to')
parser.add_argument('--use_saved_signal',
                    action='store_true',
                    default=False,
                    help="""Use --isolated_signal_path to load a signal that
                    has already been isolated from the received data.
                    Otherwise, the program will perform cross correlation on
                    the received data to isolate the signal, which can take
                    several minutes.""")
parser.add_argument('--make_plots',
                    action='store_true',
                    default=False,
                    help="Generate and display all plots")

FLAGS, unparsed = parser.parse_known_args()

# Image saving parameters.
save_name = FLAGS.figure_directory + '/{}.png'
figsize_constellation = (10, 10)
figsize_rect = (10, 5)

# Load transmission data that was created in encode.py.
tx_info = np.load(FLAGS.tx_info_path)
header = tx_info['header']
data = tx_info['data']
tx_combined = tx_info['combined']
symbols = tx_info['data_bits']

if not FLAGS.use_saved_signal:
    # load received data
    received_data = np.fromfile(FLAGS.received_data_path, dtype=np.float32)
    signal_time_rx = received_data[::2] + received_data[1::2]*1j

    plt.plot(signal_time_rx.real)
    plt.show()

    signal_len = tx_combined.shape[-1]
    lag, signal_time_rx = mimo.detect_start(signal_time_rx,
            header[0],
            signal_len, plot=FLAGS.make_plots)

    plt.plot(signal_time_rx.real)
    plt.show()

    np.savez('isolated_signal.npz', signal_rx=signal_time_rx)

else:
    # Use a signal that has previously been isolated with cross correlation.
    signal_time_rx = np.load(FLAGS.isolated_signal_path)["signal_rx"]

# Isolate the headers for timing offset correction and channel estimation.
header_1_end = header.shape[-1]
header_2_start = header.shape[-1]+mimo.ZERO_SAMPLES
header_2_end = header_2_start + header.shape[-1]
rx_header_1 = signal_time_rx[:header_1_end]
rx_header_2 = signal_time_rx[header_2_start:header_2_end]

# correct for timing differences using timing_offset/f_delta variable
f_delta_header_2, y_scale_header_2 = mimo.find_f_delta(rx_header_2)
print("f_delta from header 2: ", f_delta_header_2)

if FLAGS.make_plots:
    # Plot constellation before timing correction
    plt.figure(figsize=figsize_constellation)
    plt.plot(signal_time_rx.real[10::20], signal_time_rx.imag[10::20], ".")
    plt.title("Received Signal Constellation Before Timing Offset")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.savefig(save_name.format('received_const_before_timing'))
    plt.show()

# Correct for the timing offset.
signal_time_rx = mimo.correct_timing_offset(f_delta_header_2,
        y_scale_header_2,
        signal_time_rx)

if FLAGS.make_plots:
    plt.figure(figsize=figsize_rect)
    plt.plot(signal_time_rx)
    plt.title("Received Signal After Timing Correction")
    plt.savefig(save_name.format('received_after_timing'))
    plt.show()

    # Plot constellation after timing correction.
    plt.figure(figsize=figsize_constellation)
    plt.plot(signal_time_rx.real[:header.shape[-1]],
            signal_time_rx.imag[:header.shape[-1]], ".")
    plt.title("Received Header 1 Constellation After Timing Offset")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.savefig(save_name.format('recevied_const_after_timing'))
    plt.show()

    # Plot the constellation of header 2 after timing offset correction.
    plt.figure(figsize=figsize_constellation)
    plt.plot(signal_time_rx.real[header_2_start:header_2_end],
            signal_time_rx.imag[header_2_start:header_2_end], ".")
    plt.title("Received Header 2 Constellation After Timing Offset")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.savefig(save_name.format('received_h2_after_timing'))
    plt.show()

# Estimate the channel.
H = mimo.estimate_channel_alamouti(rx_header_1, rx_header_2, header[0], header[1])
print("channel matrix H:\n{}".format(H))

data_start = header_2_end + mimo.ZERO_SAMPLES
data_end = data_start + data.shape[-1]
rx_data = signal_time_rx[data_start:data_end]

# Downsample to account for Alamouti encoding.
downsample_start = mimo.SYMBOL_PERIOD // 2
downsample_n_samples = mimo.SYMBOL_PERIOD
rx_data_downsampled = rx_data[downsample_start::downsample_n_samples]

# Correct for the channel and recover the signal.
recovered_signal = mimo.recover_signals_alamouti(rx_data_downsampled, H) 

if FLAGS.make_plots:
    # Plot the constellation of the recovered signal.
    plt.figure(figsize=figsize_constellation)
    plt.plot(recovered_signal.real, recovered_signal.imag, ".")
    plt.title("Recovered Signal Constellation")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.savefig(save_name.format('recovered_const'))
    plt.show()

recovered_signal_symbols = mimo.turn_data_to_bits(recovered_signal)

if FLAGS.make_plots:
    # Plot per sample if the recovered bits equal the bits transmitted.
    plt.figure(figsize=figsize_rect)
    plt.plot(recovered_signal_symbols == symbols)
    plt.title("Per Sample Error")
    plt.savefig(save_name.format('errors'))
    plt.show()

print("Error rate: {}".format(mimo.calculate_error_rate(symbols, recovered_signal_symbols)))
