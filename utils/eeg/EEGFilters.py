import numpy as np
from scipy.signal import butter, cheby1, cheby2, ellip, freqz, lfilter

class EEGFilters:

    def __init__(self, fs) -> None:


        # Define filter specifications
        self.low_cutoff = 0.1  # Low cutoff frequency in Hz
        self.high_cutoff = 60.0  # High cutoff frequency in Hz
        self.fs = fs  # Sampling frequency in Hz

        # Normalize frequencies
        self.low_cutoff_norm = self.low_cutoff / (self.fs / 2)
        self.high_cutoff_norm = self.high_cutoff / (self.fs / 2)

        # Orders of Butterworth filters
        orders = [3, 4, 5]

        Butterworth_params = []

        # Plot frequency response for each order
        # plt.figure(figsize=(10, 6))
        for order in orders:
            b, a = butter(order, [self.low_cutoff_norm, self.high_cutoff_norm], btype='bandpass')
            w, h = freqz(b, a, worN=8000)
            Butterworth_params.append([(b,a), (w,h), order])
            # plt.plot(0.5 * fs * w / np.pi, np.abs(h), label="Butterworth  Order = %d" % order)
        

        # Design Chebyshev Type 1 filter
        b_cheby1, a_cheby1 = cheby1(4, 1, [self.low_cutoff_norm, self.high_cutoff_norm], btype='bandpass')

        # Design Chebyshev Type 2 filter
        b_cheby2, a_cheby2 = cheby2(4, 20, [self.low_cutoff_norm, self.high_cutoff_norm], btype='bandpass')

        # Design elliptic filter
        b_ellip, a_ellip = ellip(4, 1, 20, [self.low_cutoff_norm, self.high_cutoff_norm], btype='bandpass')

        # Frequency response calculation
        w, h_cheby1 = freqz(b_cheby1, a_cheby1)
        w, h_cheby2 = freqz(b_cheby2, a_cheby2)
        w, h_ellip = freqz(b_ellip, a_ellip)