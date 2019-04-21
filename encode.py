"""This script contains code to generate and encode data to transmit using
Alamouti space-time block coding.

Authors: Annie Ku, Ariana Olson
"""
import numpy as np
import utils.mimo_util as mimo

# Generate random QPSK data and headers.
seed = 5
symbols_data = mimo.generate_symbols(mimo.DATA_SYMBOLS, seed)
symbols_header = mimo.generate_symbols(mimo.HEADER_SYMBOLS, seed + 1)

# Modulate with PAM.
tx_data = mimo.generate_tx_data_2x2(symbols_data, mimo.SYMBOL_PERIOD)
header_data = mimo.generate_tx_data_2x2(symbols_header, mimo.SYMBOL_PERIOD)

# Combine data and save to files for USRP transmission.
tx_combined = mimo.add_headers_2x2(tx_data, header_data, mimo.ZERO_SAMPLES)
mimo.interleave_and_save_data(tx_combined, 'tx_1.dat', 'tx_2.dat')

# Save data arrays for decoding and BER calculations.
np.savez('tx_info.npz', header=header_data, data=tx_data, combined=tx_combined)
