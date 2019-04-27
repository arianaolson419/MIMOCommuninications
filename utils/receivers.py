"""This file contains functions to implement the receivers in both Part a and
Part b of the Principles of Wireless Communications Lab 2
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

def detect_start_lts(signal_time_rx, lts, signal_length):
    cross_corr = np.correlate(signal_time_rx, lts)
    lag = np.argmax(np.abs(cross_corr)) - 1

    plt.plot(cross_corr)
    plt.title("Corr")
    plt.show()

    print(lag)

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
    """Given 2 two transmit antennas & 1 receive, make a channel matrix like this:
        [[h1, h2]
        [h*_2, -h*_1]]
    
    Arguments:
        rx_sections {[type]} -- [description]
        tx_headers {[type]} -- [description]
    """
    # TODO: parse the tx and rx sections accordingly!
    #h1 = estimate_channel_mimo(rx_sections[], tx_headers[])
    #h2 = estimate_channel_mimo(rx_sections[], tx_headers[])
    # TODO: return matrix from format in function's docstring
    H = estimate_channel(rx_header_1, rx_header_2, tx_header_1, tx_header_2)
    H_alamouti = np.zeros((2, 2), dtype=np.complex128)
    H_alamouti[0][0] = H[0]
    H_alamouti[0][1] = H[1]
    H_alamouti[1][0] = np.conj(H[1])
    H_alamouti[1][1] = -np.conj(H[0])

    return H_alamouti

def calculate_weights_zero_forcing(H):
    """Calculates the weight matrix for the zero-forcing receiver.

    Args:
        H (complex (2, 2) ndarray): A matrix of channel estimations.

    Returns:
        W (complex (2, 2) ndarray): A matrix of weights.
    """
    W = np.linalg.inv(H)
    return W

def calculate_weights_mmse(tx_power, sigma, H):
    """Calculates the weight vectors for the MMSE receiver.

    Args:
        tx_power (float): The power of the transmitted signal.
        H (complex (2, 2) ndarray): A matrix of channel estimations.
        rx1 (complex 1D ndarray): The signal received at rx antenna 1.
        rx2 (complex 1D ndarray): The signal received at rx antenna 2.

    Returns: 
        w1 (complex 1D ndarray): The weight vector for recovering rx1.
        w2 (complex 1D ndarray): The weight vector for recovering rx2.
    """
    # Calculate R.
    h1 = H[0, :]
    h2 = H[1, :]

    R = tx_power * h1 * np.transpose(np.conjugate(h1)) + tx_power * h2 \
        * np.transpose(np.conjugate(h2)) + sigma * np.eye(2)

    # Calculate the weight matrix
    W = tx_power * np.linalg.inv(R) * H

    return W


def recover_signals(rx1, rx2, W):
    """Estimates the sent signals using the weight matrix.

    Args:
        rx1 (complex 1D ndarray): The signal received at rx antenna 1.
        rx2 (complex 1D ndarray): The signal received at rx antenna 2.
        W (complex (2, 2) ndarray): A matrix of weights.

    Returns:
        x1_est (complex 1D ndarray): The estimated signal transmitted from
            tx antenna 1.
        x2_est (complex 1D ndarray): The estimated signal transmitted from
            tx antenna 2.
    """
    y1 = np.copy(rx1)
    y2 = np.copy(rx2)

    ys = np.vstack((y1, y2))
    x_est = np.matmul(W, ys)

    x1_est = np.squeeze(x_est[0, :])
    x2_est = np.squeeze(x_est[1, :])

    return x1_est, x2_est

def recover_signals_mimo(rx, W):
    """Use a weight matrix to recover MIMO signals.
    Args:
        rx (complex (4, num_samples) ndarray): The received MIMO signals with
            each row as the received signal at one antenna.
        W (complex (4, 4) ndarray): A matrix of weights to apply to the signal.

    Returns:
        x_est (complex (4, num_samples) ndarray): The recovered MIMO signals in
            the same format as rx.
    """
    return np.matmul(W, rx)

def preprocess_tx(tx, H):
    """Transform the transmitted data with known channel information.

    Args:
        tx (complex (4, n) ndarray): The n sample long signals to transmit.
        H (complex (4, 4) ndarray): A matrix of channel estimations.

    Returns:
        U (complex (4, 4) ndarray): The unitary matrix U resulting from
            performing SVD on the channel estimations.
        S (complex (4, 4) ndarray): The diagonal matrix S representing the
            singular values of the SVD.
        tx_transform: The signal to transmit that's been transformed using CSI.
    """
    U, S, Vh = np.linalg.svd(H)
    tx_transform = np.matmul(np.transpose(np.conjugate(Vh)), tx)
    return U, S, tx_transform

def recover_signals_csi(rx, U):
    """Recover signals transformed with known CSI.

    Args:
        rx (complex (4, n) ndarray: The n sample long received signals.
        U (complex (4, 4) ndarray: The unitary matrix U from the SVD of the channel estimates.

    Returns:
        s_est: The recovered transmitted symbols.
    """
    s_est = np.matmul(np.transpose(np.conjugate(U)), rx)
    return s_est

def estimate_f_delta(lts, num_samples):
    """Estimate the frequency offset of a received OFDM signal using the LTS.

    Args:
        lts (1D complex ndarray): A block of random complex values taking on
            values of +-1 +-j of length 3 * num_samples that has been sent through a nonflat channel. This contains 3 repeating
            blocks of size num_samples.
        num_samples (int): The number of samples in each LTS block.

    Returns:
        f_delta_est (float): The average estimated frequency offset, in radians.
    """
    sum_f_delta_ests = 0
    for i in range(num_samples):
        complex_exp = lts[2 * num_samples + i] / lts[num_samples + i]
        sum_f_delta_ests += np.angle(complex_exp)

    return sum_f_delta_ests / (num_samples ** 2)



def detect_start(signal_time_rx, header, signal_length):
    """Find the start of the time domain signal using crosss correlation.

    Args:
        signal_time_rx (1D complex ndarray): A received time domain signal
            that conatains a known header as the first portion of the signal.
        header (1D complex ndarray): A known signal that is included at the
            beginning of the transmitted signal.
        signal_length (int): The length of the signal that whas transmitted.

    Returns:
        signal_time_start (1D complex ndarray): A version of signal_time_rx
            that starts at the first data sample.
    """
    cross_corr = np.correlate(signal_time_rx, header)
    lag = np.argmax(cross_corr) - 1

    return signal_time_rx[lag:lag+signal_length]

def correct_freq_offset(signal_time, f_delta):
    """Correct for the frequency offset in a time-domain signal.

    Args:
        signal_time (1D complex ndarray): A time domain signal that starts with
            the LTS used to estimate the frequency offset.
        f_delta (float): The frequency offset to correct for, in radians.

    Returns:
        signal_time_corrected (1D complex ndarray): The corrected time domain
            signal. Has the same shape as signal_time.
    """
    exponentials = np.exp(np.arange(signal_time.shape[-1]) * 1j * f_delta)
    signal_time_corrected = signal_time / exponentials

    return signal_time_corrected


def decode_signal_freq(channel_coefficients, signal_freq):
    """Decode a frequency domain signal into a series of bits.

    Args:
        signal_freq (1D float32 ndarray): The bit sequence in the frequency
            domain.

    Returns:
        bits (1D ndarray): The bit sequence decoded from signal_freq.
    """
    # TODO: Find the symbol that's closest to symbol 0
    # h1 = 
    # h2 = 
    # h1_conjugate = 
    # h2_conjugate = 

    bits = np.sign(signal_freq.real) + 1j * np.sign(signal_freq.imag)

    return bits


