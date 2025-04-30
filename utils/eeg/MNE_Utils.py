import mne
from mne.preprocessing import ICA
from mne.filter import filter_data

class MNE_Utils:
    def __init__(self) -> None:
        pass

    def createInfoForDataset(self, channel_names, channel_types, sampling_freq=1000):
        return mne.create_info(ch_names=channel_names, ch_types=channel_types, sfreq=sampling_freq)

    def create_RAW_data(self, eeg_features, channel_names, channel_types,sampling_freq=1000):

        numberofSamples,freqLength,numChannels = eeg_features.shape

        # sfreq = sampling_freq  # Replace with your actual sampling frequency
        # channel_types = ['eeg'] * len(channel_names)
        # info = mne.create_info(ch_names=channel_names, ch_types=channel_types, sfreq=sfreq)

        info = self.createInfoForDataset(channel_names,channel_types, sampling_freq)

        # Create MNE raw data object from NumPy array
        RawSamples = []
        for samplesIdx in range(numberofSamples):
            raw_data = mne.io.RawArray(eeg_features[samplesIdx,:,:].T, info)  # Transpose EEG_data to match MNE's channel x time format
            RawSamples.append(raw_data)
        Raw_MNE_EEG_DATA = mne.io.concatenate_raws(RawSamples)

        return Raw_MNE_EEG_DATA
    


    def filter_frequency_bands(self,Raw_MNE_EEG_DATA,Lf,Hf,sampling_freq=1000):

        """
        # Filter Theta, beta and gamma bands
        Delta => 0.5 - 4 Hz
        Theta => 4 - 8 Hz
        Alpha => 8-14 Hz
        Beta  => 14-30 Hz
        Gamma => >30 Hz
        """
        NumpyRaw = Raw_MNE_EEG_DATA.get_data()
        filtered_eeg = filter_data(NumpyRaw,sampling_freq,l_freq=Lf,h_freq=Hf) # Include Alpha, beta and gamma bands
        raw_data = mne.io.RawArray(filtered_eeg, Raw_MNE_EEG_DATA.info) 
        Raw_MNE_EEG_DATA = mne.io.concatenate_raws([raw_data])

    
        filt_raw = Raw_MNE_EEG_DATA.copy().filter(l_freq=Lf, h_freq=Hf)
        # mne.channels.get_builtin_montages()
        for builtInMontage in mne.channels.get_builtin_montages():
            Montage  = mne.channels.make_standard_montage(kind=builtInMontage)
            try:
                filt_raw.set_montage(Montage)
                print(f"set montage done for  {Montage}")
                break
            except:
                pass

        return filt_raw
    

    def checkFrequencies(self, Raw_MNE_EEG_DATA):

        spectrum = Raw_MNE_EEG_DATA.compute_psd(method="welch", fmax=400)
        psd, freqs = spectrum.get_data(return_freqs=True)

        # Now, you can check the power in various frequency bands
        # Delta (0.5–4 Hz)
        delta = psd[:, (freqs >= 0.5) & (freqs <= 4)].mean(axis=-1)

        # Theta (4–7 Hz)
        theta = psd[:, (freqs >= 4) & (freqs <= 7)].mean(axis=-1)

        # Alpha (8–13 Hz)
        alpha = psd[:, (freqs >= 8) & (freqs <= 13)].mean(axis=-1)

        # Beta (13–30 Hz)
        beta = psd[:, (freqs >= 13) & (freqs <= 30)].mean(axis=-1)

        # Gamma (30–140 Hz)
        gamma = psd[:, (freqs > 30) & (freqs <= 140)].mean(axis=-1)

        # plt.figure(figsize=(20, 4))
        # plt.loglog(freqs, psd.T)
        # plt.title('PSD (dB) of EEG channels')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Power Spectral Density (dB)')
        # plt.show()

        print('Delta power: ', delta.mean())
        print('Theta power: ', theta.mean())
        print('Alpha power: ', alpha.mean())
        print('Beta power: ', beta.mean())
        print('Gamma power: ', gamma.mean())