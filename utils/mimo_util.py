"""This file contains functions used to implement MIMO in hardware with
Alamouti space-time coding.

Authors: Annie Ku, Ariana Olson
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

# Constants used for signal creation and analysis.
SYMBOL_PERIOD = 20 # samples
HEADER_SYMBOLS = 100
DATA_SYMBOLS = 1000
ZERO_SAMPLES = 5000 # samples
NUM_CHANNELS = 2

def generate_symbols(num_symbols, seed):
    """Generate a single channel of random complex data symbols

    Args:
        num_symbols (int): The number of symbols per channel.
        seed (int): The seed used by the random number generator.

    Returns:
        symbols (complex 1D ndarray): An array of size (num_symbols,) symbols of
        +/-1 +/-1j.
    """
    all_symbols = np.zeros((num_symbols))
    np.random.seed(seed)
    symbols_real = np.sign(np.random.randn(num_symbols))
    symbols_imag = np.sign(np.random.randn(num_symbols)) * 1j

    symbols = symbols_real + symbols_imag
    return symbols

def generate_header_data(symbols, symbol_period):
    """Generate data from an array of complex symbols to transmit over USRP.
    Symbols are modulated with rectangular pulses. This data is not alamouti
    encoded.

    Args:
        symbols (complex 1d ndarray): An array of symbols of values +/-1 and
            +/-1j.
        symbol_period (int): The number of samples in a single pulse.

    Returns:
        header_data (1d complex ndarray): An array of
            shape(symbols.shape[-1] * symbol_period,) representing the data of
            one header.
    """
    pulse = np.ones(symbol_period)
    x = np.zeros((symbols.shape[-1] * symbol_period), dtype=np.complex128)
    x[::symbol_period] = symbols
    header = np.convolve(x, pulse)[:symbols.shape[-1] * symbol_period]

    return header

def generate_tx_data_2x2(symbols, symbol_period):
    """Generate Alamouti encoded QPSK data from an array of complex symbols to
    transmit over USRP. Symbols are modulated with rectangular pulses.

    Args:
        symbols (complex 1d ndarray): An array of symbols of values +/-1 and
            +/-1j.
        symbol_period (int): The number of samples in a single pulse.

    Returns:
        tx (2D complex ndarray): An array of shape
            (2, symbols.shpae[-1] * symbol_period) representing the data to
            transmit from both antennas. The first row contains data to transmit
            from the first antenna and the second row contains data to transmit
            from the second antenna.
    """

    # Encode the symbols with the Alamouti scheme.
    alamouti = np.zeros((2, symbols.shape[-1]), dtype=np.complex128)
    alamouti[0, ::2] = symbols[::2]
    alamouti[1, ::2] = symbols[1::2]

    alamouti[0, 1::2] = -np.conj(symbols[1::2])
    alamouti[1, 1::2] = np.conj(symbols[::2])

    # Modulate for QPSK.
    pulse = np.ones(symbol_period)
    tx1 = np.zeros((2, alamouti.shape[-1] * symbol_period), dtype=np.complex128)
    tx1[0, ::symbol_period] = alamouti[0].real
    tx1[1, ::symbol_period] = alamouti[0].imag

    tx2 = np.zeros((2, alamouti.shape[-1] * symbol_period), dtype=np.complex128)
    tx2[0, ::symbol_period] = alamouti[1].real
    tx2[1, ::symbol_period] = alamouti[1].imag

    tx1[0] = np.convolve(tx1[0], pulse)[:alamouti.shape[-1] * symbol_period]    # real
    tx1[1] = np.convolve(tx1[1], pulse)[:alamouti.shape[-1] * symbol_period]    # imag

    tx2[0] = np.convolve(tx2[0], pulse)[:alamouti.shape[-1] * symbol_period]    # real
    tx2[1] = np.convolve(tx2[1], pulse)[:alamouti.shape[-1] * symbol_period]    # imag

    tx = np.zeros((2, tx1.shape[-1]), dtype=np.complex128)

    # Antenna 1
    tx[0] = tx1[0] + tx1[1] * 1j

    # Antenna 2
    tx[1] = tx2[0] + tx2[1] * 1j

    return tx

def add_headers_2x2(tx_data, headers, zero_samples, scale=1.0):
    """Combine headers and data for transmission.

    the order of transmission is as follows:
    1. Both antennas transmit zero_samples number of zeros.
    2. Antenna 1 transmits header_1 and antenna 2 transmits the same number of
        zeros.
    3. Both antennas transmit zero_samples number of zeros.
    4. Antenna 2 transmits header_2 and antenna 1 transmits the same number of
        zeros
    5. Both antennas transmit zero_samples number of zeros.
    6. Antenna 1 transmits tx_1 and antenna 2 transmits tx_2.

    Args:
        tx_data (complex ndarray): A two-row array with each row representing
            the data to transmit from one antenna.
        headers (complex ndarray): A two-row array with each row representing
            the header to transmit from one antenna.
        zero_samples (int): The number of zero samples to pad between headers
            and data.
        scale (float): Amplitude scale factor between 0 and 1.
    
    Returns:
        tx_combined (complex ndarray): A two-row array containing the full
            transmission sequence for each antenna. Each row represents the
            sequence for one antenna.
    """
    header_1_start = zero_samples
    header_1_end = header_1_start + headers.shape[-1]

    header_2_start = header_1_end + zero_samples
    header_2_end = header_2_start + headers.shape[-1]

    data_start = header_2_end + zero_samples

    tx_size = 3 * zero_samples + headers.shape[-1] * 2 + tx_data.shape[-1]
    tx_combined = np.zeros((2, tx_size), dtype=np.complex128)

    tx_combined[0, header_1_start:header_1_end] = headers[0]
    tx_combined[1, header_2_start:header_2_end] = headers[1]
    tx_combined[:, data_start:] = tx_data

    return tx_combined * scale

def interleave_and_save_data(tx_combined, dest_path_tx1, dest_path_tx2):
    """Create data files to transmit over USRP.

    Args:
        tx_combined (complex 2D ndarray): A two-row array containing the full
            transmission sequence for each antenna, including headers. Each row
            represents the sequence for one antenna.
        dest_path_tx_1 (string): The path to save the sequence for antenna 1 to.
        dest_path_tx2 (string): The path to save the sequence for antenna 2 to.
    """
    tmp = np.zeros((2, tx_combined.shape[-1] * 2), dtype=np.float32)
    tmp[:, ::2] = tx_combined.real
    tmp[:, 1::2] = tx_combined.imag

    tmp[0, :].tofile(dest_path_tx1)
    tmp[1, :].tofile(dest_path_tx2)

def detect_start(signal_time_rx, header, signal_length, plot=False):
    """Detect the start of the signal from received data using cross correlation.

    Args:
        signal_time_rx (complex 1D ndarray): A time domain signal of receieved
            data containing the signal.
        header (complex 1D ndarray): The header to use for cross correlation.
        signal_length (int): The number of samples in the signal to isolate.
        plot (bool): If True, plot the cross correlation of the signal and the
            header. Default is False.

    Returns:
       lag (int): The index of the first sample of the signal in the received
           data.
       signal_time_rx (complex 1D ndarray): The time domain signal isolated
           from the received data. Its shape is (signal_len,)
    """
    cross_corr = np.correlate(signal_time_rx, header)
    lag = np.argmax(np.abs(cross_corr)) - 1

    if plot:
        plt.plot(cross_corr)
        plt.title("corr")
        plt.show()

    print('start of signal: {}'.format(lag))

    return lag, signal_time_rx[lag:lag+signal_length]

def estimate_channel(rx_header_1, rx_header_2, tx_header_1, tx_header_2):
    """Estimate the channel using the headers sent from each antenna.
    Channel estimation is the same for both the zero-forcing and MMSE
    receivers. This function uses copies of the arrays given to it because any
    0s are replaced by small numbers ot avoid division errors.
    Args:
        rx_header_1 (complex 1D ndarray): The portion of the signal received at
            rx antenna 1 corresponding to the header transmitted at tx antenna
            1.
        rx_header_2 (complex 1D ndarray): The portion of the signal received at
            rx antenna 2 corresponding to the header transmitted at tx antenna
            2.
        tx_header_1 (complex 1D ndarray): The header transmitted from tx antenna
            1.
        tx_header_2 (complex 1D ndarray): The header transmitted from tx antenna
            2.
    Returns:
        H (complex (2,) ndarray): A matrix of channel estimations.
    """
    header11 = np.copy(rx_header_1)
    header12 = np.copy(rx_header_2)

    # Replace 0s in denominators to avoid division errors.
    tx_header_1[tx_header_1 == 0] = 1e-12
    tx_header_2[tx_header_2 == 0] = 1e-12

    H = np.zeros(2, dtype=np.complex128)
    H[0] = np.mean(header11 / tx_header_1)
    H[1] = np.mean(header12 / tx_header_2)

    return H

def estimate_channel_alamouti(rx_header_1, rx_header_2, tx_header_1, tx_header_2):
    """Estimate the channel for a 2X1 MIMO system with Alamouti encoding. The channel estimation matrix is as follows:
       H = [[h1, h2]
           [h*_2, -h*_1]]
    Where h1 is the channel from the first transmit antenna to the receive
    antenna, and h2 is the channel from the second transmit antenna to the
    receive antenna.
    
    Args:
        rx_header_1 (complex 1D ndarray): The header received from the first
            antenna.
        rx_header_2 (complex 1D ndarray): The header received from the second
            antenna.
        tx_header_1 (complex 1D ndarray): The header transmitted from the first
            antenna.
        tx_header_2 (complex 1D ndarray): The header transmitted from the
            second antenna.
    """
    H = estimate_channel(rx_header_1, rx_header_2, tx_header_1, tx_header_2)
    H_alamouti = np.zeros((2, 2), dtype=np.complex128)
    H_alamouti[0][0] = H[0]
    H_alamouti[0][1] = H[1]
    H_alamouti[1][0] = np.conj(H[1])
    H_alamouti[1][1] = -np.conj(H[0])

    return H_alamouti

def split_signal(ys):
    """Split a one-dimensional signal into two rows in which each column
    contains a symbol transmitted at one time and a signal transmitted at the
    subsequent time for Alamouti decoding.
    
    Args:
        ys (1D ndarray): An array of samples to split into two rows of data.
    
    Returns:
        split (2D ndarray): A two-row array in which the first row contains
            every other sample of ys starting at the first sample and the
            second row contains the remaining samples.
    """
    split = np.zeros((2, ys.shape[-1] // 2), dtype=np.complex128)
    split[0] = ys[::2]
    split[1] = ys[1::2]
    return split

def merge_signal(split):
    """Undoes the split_signal function.

    For a 1D array ys of shape (1, N), N % 2 = 0,
    merge_signal(split_signal(ys)) = ys.

    Args:
        split (2D ndarray): A two row array to merge into a one-dimensional
            array.

    Returns:
        merge (1D ndarrys): A one dimensional array containing interleaved
            samples of the rows of split, starting with split[0][0].
    """
    merge = np.zeros(split.shape[-1] * 2, dtype=np.complex128)
    merge[::2] = split[0]
    merge[1::2] = split[1]
    return merge

def recover_signals_alamouti(rx_data, H):
    """Recovers an Alamouti encoded signal using a zero-forcing receiver.

    Args:
        rx_data (complex 1D ndarray): A downsampled Alamouti encoded signal to
            recover.
        H (complex (2, 2) ndarray): An array of channel estimations.

    Returns:
        recovered_merged (complex 1D ndarray): The recovered signal.
    """
    rx_split = split_signal(rx_data)
    rx_split[1] = np.conj(rx_split[1])
    
    recovered = np.matmul(np.linalg.pinv(H), rx_split)
    
    recovered_merged = merge_signal(recovered)
    
    return recovered_merged

def calculate_error_rate(recovered_signal_bits, transmitted_signal_bits):
    """decode a frequency domain signal into a series of bits.

    args:
        signal_freq (1d float32 ndarray): the bit sequence in the frequency
            domain.

    returns:
        bits (1d ndarray): the bit sequence decoded from signal_freq.
    """
    
    return np.sum(recovered_signal_bits != transmitted_signal_bits)/recovered_signal_bits.shape[-1]

def turn_data_to_bits(recovered_symbols):
    """Use the sign of recovered samples to create symbols of +/-1 +/-1j for
    bit error rate calculations.

    Args:
        recovered_symbols (complex 1D ndarray): An array of complex samples.

    Returns:
        symbols (complex 1D ndarray): An array of the same shape as
            recovered_symbols containing values of +/-1 +/-1j.
    """
    symbols = np.sign(recovered_symbols.real) + 1j * np.sign(recovered_symbols.imag)
    return symbols

def find_f_delta(header):
    """Estimate the frequency offset of a received QPSK signal.

    Args:
        header (complex 1D ndarray): A non-Alamouti encoded QPSK header to use
            to estimate the frequency offset.
    Returns:
        f_delta (complex float): The frequecy offset of the received signal.
        y_scale (complex_float): The magnitude of the maximum value of the fft.
            This is used for correcting for the timing offset.
    """
    # Normalize.
    h_mag_est = np.sqrt(np.mean(np.square(header)))
    header_normalized = header / h_mag_est

    # Create s[k] by raising the signal to the 4th power.
    s = np.power(header_normalized, 4)
    fft = np.fft.fft(s)
    shifted_fft = np.fft.fftshift(fft)
    freq_axis = np.linspace(-np.pi, np.pi - ((2 * np.pi)/(shifted_fft.shape[-1]+1)), shifted_fft.shape[-1])

    # Get f_delta and theta from finding frequency value and peak height.
    x_offset = freq_axis[np.argmax(np.abs(shifted_fft))]
    y_height = np.max(shifted_fft)
    y_scale = y_height / np.abs(y_height)
    f_delta  = (x_offset) / -4 

    return f_delta, y_scale

def correct_timing_offset(f_delta, y_scale, signal_rx_time):
    """Correct for timing offsets in a received signal.
    
    Args:
        f_delta (complex float): The estimated frequency offset.
        signal_rx_time (complex 1D ndarray): The entire time domain signal, including
        both headers and data.

    Returns:
        x_est (complex 1D ndarray): The timing-offset corrected signal of the
            same shape as signal_rx_time.
    """
    psi = f_delta * np.arange(0,signal_rx_time.shape[-1])
    x_est = signal_rx_time * np.exp(1j * psi) / np.power(y_scale, 0.25)

    return x_est
