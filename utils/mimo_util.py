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
        num_channels (int): The number of channels of data.
        seed (int): The seed used by the random number generator.

    Returns:
        symbols (complex 1D ndarray): An array of size (num_channels, num_symbols)
            symbols of +/-1 +/-1j.
    """
    all_symbols = np.zeros((num_symbols))
    np.random.seed(seed)
    symbols_real = np.sign(np.random.randn(num_symbols))
    symbols_imag = np.sign(np.random.randn(num_symbols)) * 1j

    symbols = symbols_real + symbols_imag
    return symbols

def generate_header_data(symbols, symbol_period):
    """Generate data from an array of complex symbols to transmit over USRP.
    Symbols are modulated with rectangular pulses. This data is not Alamouti
    encoded.

    Args:
        symbols (complex 1D ndarray): An array of length num_symbols
            symbols of +/-1 and +/-1j.
        symbol_period (int): The number of samples in a single pulse.

    Returns:
        header_data (1D complex ndarray): An array representing the data of one header.
    """
    pulse = np.ones(symbol_period)
    x = np.zeros((symbols.shape[-1] * symbol_period), dtype=np.complex128)
    x[::symbol_period] = symbols
    header = np.convolve(x, pulse)

    header_data = np.zeros((2 * header.shape[-1]))
    header_data[::2] = header.real
    header_data[1::2] = header.imag

    return header_data

def generate_tx_data_2x2(symbols, symbol_period):
    """Generate alamouti encoded data from an array of complex symbols to
    transmit over USRP. Symbols are modulated with rectangular pulses.

    Args:
        symbols (complex 1D ndarray): An array of length num_symbols
            symbols of +/-1 and +/-1j.
        symbol_period (int): The number of samples in a single pulse.

    Returns:
        tx_1 (1D complex ndarray): An array representing the data to transmit
            from the first antenna
        tx_2 (1D complex ndarray): An array representing the data to transmit
            from the second antenna
    """
    pulse = np.ones(symbol_period)
    x = np.zeros((2, symbols.shape[-1] * symbol_period), dtype=np.complex128)
    tmp = np.zeros((2, symbols.shape[-1]), dtype=np.complex128)

    # Encode with Alamouti scheme.
    for i in range(0, symbols.shape[-1], 2):
        tmp[0, i] = symbols[i]
        tmp[0, i + 1] = symbols[i + 1]
        tmp[1, i] = -np.conjugate(symbols[i + 1])
        tmp[1, i + 1] = np.conjugate(symbols[i])

    # Split into real and imaginary components.
    x[:, ::symbol_period] = tmp
    
    # Modulate with PAM
    tx1 = np.convolve(x[0], pulse)
    tx2 = np.convolve(x[1], pulse)

    tx_data = np.zeros((2, 2 * tx1.shape[-1]), dtype=np.complex128)
    tx_data[0, ::2] = tx1.real
    tx_data[0, 1::2] = tx1.imag
    tx_data[1, ::2] = tx2.real
    tx_data[1, 1::2] = tx2.imag

    return tx_data

def add_headers_2x2(tx_data, headers, zero_samples):
    """Combine headers and data for transmission.

    The order of transmission is as follows:
    1. Both antennas transmit zero_samples number of zeros.
    2. Antenna 1 transmits header_1 and antenna 2 transmits the same number of
        zeros.
    3. Both antennas transmit zero_samples number of zeros.
    4. Antenna 2 transmits header_2 and antenna 1 transmits the same number of
        zeros
    5. Both antennas transmit zero_samples number of zeros.
    6. Antenna 1 transmits tx_1 and Antenna 2 transmits tx_2.

    Args:
        tx_data (complex ndarray): A two-row array with each row representing
            the data to transmit from one antenna.
        headers (complex ndarray): A two-row array with each row representing
            the header to transmit from one antenna.
        zero_samples (int): The number of zero samples to pad between headers
            and data.
    
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

    return tx_combined

def interleave_and_save_data(tx_combined, dest_path_tx1, dest_path_tx2):
    """Interleave the real and imaginary components of the transmission sequences

    Args:
        tx_combined (complex ndarray): A two-row array containing the full
            transmission sequence for each antenna, including headers. Each row
            represents the sequence for one antenna.
        dest_path_tx_1 (string): The path to save the sequence for antenna 1 to.
        dest_path_tx2 (string): The path to save the sequence for antenna 2 to.
    """
    tmp = np.zeros((2, tx_combined.shape[-1] * 2))
    tmp[:, ::2] = tx_combined.real
    tmp[:, 1::2] = tx_combined.imag

    tmp[0, :].tofile(dest_path_tx1)
    tmp[1, :].tofile(dest_path_tx2)
