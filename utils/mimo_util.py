"""This file contains functions used to implement MIMO in hardware with
Alamouti space-time coding.

Authors: Annie Ku, Ariana Olson
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

# Constants used for signal creation and analysis.
SYMBOL_PERIOD = 20 # samples
HEADER_SYMBOLS = 200
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
    header = np.convolve(x, pulse)[:symbols.shape[-1] * symbol_period]

    return header

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

    # Encode the symbols with the Alamouti scheme.
    alamouti = np.zeros((2, symbols.shape[-1]), dtype=np.complex128)
    alamouti[0, ::2] = symbols[::2]
    alamouti[1, ::2] = symbols[1::2]

    alamouti[0, 1::2] = -np.conj(symbols[1::2])
    alamouti[1, 1::2] = np.conj(symbols[::2])

    pulse = np.ones(symbol_period)
    tx1 = np.zeros((2, alamouti.shape[-1] * symbol_period), dtype=np.complex128)
    tx1[0, ::symbol_period] = alamouti[0].real
    tx1[1, ::symbol_period] = alamouti[0].imag

    tx2 = np.zeros((2, alamouti.shape[-1] * symbol_period), dtype=np.complex128)
    tx2[0, ::symbol_period] = alamouti[1].real
    tx2[1, ::symbol_period] = alamouti[1].imag

    # Modulate for QPSK.
    tx1[0] = np.convolve(tx1[0], pulse)[:alamouti.shape[-1] * symbol_period]    # Real
    tx1[1] = np.convolve(tx1[1], pulse)[:alamouti.shape[-1] * symbol_period]    # Imag

    tx2[0] = np.convolve(tx2[0], pulse)[:alamouti.shape[-1] * symbol_period]    # Real
    tx2[1] = np.convolve(tx2[1], pulse)[:alamouti.shape[-1] * symbol_period]    # Imag

    tx = np.zeros((2, tx1.shape[-1]), dtype=np.complex128)

    # Antenna 1
    tx[0] = tx1[0] + tx1[1] * 1j

    # Antenna 2
    tx[1] = tx2[0] + tx2[1] * 1j

    return tx

def add_headers_2x2(tx_data, headers, zero_samples, scale=1.0):
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
    """Interleave the real and imaginary components of the transmission sequences

    Args:
        tx_combined (complex ndarray): A two-row array containing the full
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
