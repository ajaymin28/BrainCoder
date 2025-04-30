import numpy as np
from sklearn.decomposition import FastICA
import numpy as np
from scipy import signal
import torch
import matplotlib.pyplot as plt

class Utilities:
    def __init__(self) -> None:
        pass

    def read_channel_map(self, input_file):
        lines = []
        channel_map = {}
        with open(input_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                line = line.split("=")
                SciTerm = line[-1]
                ChanNum = line[0]
                ChanNum = ChanNum.split("-")
                ChanNum_int = int(ChanNum[-1])
                ChanNum = ChanNum[0]
                channel_map[ChanNum_int] = SciTerm
        return channel_map
    

    def load_data_label_wise(self, dataset, model,CustModel,device,process_data_with_model=False):
        label_wise_data = {}
        for data in dataset:
            eeg, label, image,i = data
            test_eeg = None

            imagePath = dataset.getImagePath(i)

            if process_data_with_model:
                with torch.no_grad():
                    features = model(image.unsqueeze(0).to(device))
                    features = features.view(-1, features.size(1))
                    outputs = CustModel(features.to(device))
                test_eeg = outputs.cpu().numpy() 

            if not label["ClassId"] in label_wise_data:
                label_wise_data[label["ClassId"]] = {"images":[], "eeg":[], "pred_eeg":[]}

            label_wise_data[label["ClassId"]]["images"].append(imagePath)
            label_wise_data[label["ClassId"]]["eeg"].append(eeg)
            label_wise_data[label["ClassId"]]["pred_eeg"].append(test_eeg)
        return label_wise_data
    

    def prepareEEGData(self, labelWiseData, convert_to_numpy=True, flatten_eeg=True, isModelPreprocessedData=False):
        eeg_features_ = []
        eeg_labels_ = []
        for label, labeData in labelWiseData.items():
            if isModelPreprocessedData:
                pred_eeg_fet = labeData["pred_eeg"]
            else:
                pred_eeg_fet = labeData["eeg"]
            for eeg in pred_eeg_fet:
                eeg_features_.append(eeg)
                eeg_labels_.append(label)
        if convert_to_numpy:
            eeg_features_  =np.array(eeg_features_, dtype=float)
        if flatten_eeg:
            eeg_features_ = eeg_features_.reshape(eeg_features_.shape[0], -1) 
        return eeg_features_, eeg_labels_
    

    def PrepapreDataforVis(self,dataset):

        label_wise_data = {}
        for data in dataset:
            eeg, label, image, i = data
            if not label["ClassId"] in label_wise_data:
                label_wise_data[label["ClassId"]] = {"images":[], "eeg":[]}

            label_wise_data[label["ClassId"]]["images"].append(image)
            label_wise_data[label["ClassId"]]["eeg"].append(eeg.numpy())
            # print(eeg.numpy().shape)

        # Exclude_labels = [0,22,15,14,16]
        Exclude_labels = []

        eeg_features = []
        eeg_labels = []
        for label, labeData in label_wise_data.items():
            eeg_fet = labeData["eeg"]
            for eeg in eeg_fet:
                # if label in Exclude_labels:
                #     continue
                # if label==12 or label==20:
                    # continue
                eeg_features.append(eeg)
                eeg_labels.append(label)
        eeg_features  =np.array(eeg_features, dtype=float)

        return label_wise_data, eeg_features, eeg_labels
    


    def remove_noise(self,eeg_data, sampling_rate):
        # Define filter parameters
        nyquist_freq = 0.5 * sampling_rate
        low_cutoff = 1.0  # Define your low cutoff frequency
        high_cutoff = 50.0  # Define your high cutoff frequency
        filter_order = 4  # Define your filter order

        # Calculate filter coefficients
        low = low_cutoff / nyquist_freq
        high = high_cutoff / nyquist_freq
        b, a = signal.butter(filter_order, [low, high], btype='band')

        # Apply the filter to each EEG channel
        filtered_data = np.zeros_like(eeg_data)
        for s in range(eeg_data.shape[0]):
            for i in range(eeg_data.shape[-1]):  # assuming eeg_data is (samples,time_points, channels)
                filtered_data[s,:, i] = signal.filtfilt(b, a, eeg_data[s,:, i])
        return filtered_data

    def remove_noise_with_ica(self,eeg_data,n_components=20):
        # Initialize ICA model
        
        # Reshape EEG data to have shape (samples, channels, time_points)
        num_samples, time_points, num_channels = eeg_data.shape

        denoised_zeros = np.zeros(shape=(num_samples, num_channels, n_components), dtype=float)
        
        for i in range(num_samples):
            ica = FastICA(n_components=n_components, random_state=42, max_iter=300)
            # print("fitting: ", eeg_data[i,:, :].shape)
            ica.fit(eeg_data[i,:, :].T)
            denoised_data = ica.transform(eeg_data[i,:, :].T)
            # print("transformed: ", denoised_data.shape)
            denoised_zeros[i,:,:] = denoised_data

        print(denoised_zeros.shape)

        return denoised_zeros
    

    def plotSampleEEGChannels(self, eeg_data, channels_to_plot):

        plt.clf()
        plt.figure().set_size_inches(20,5)
        # for x in range(normalized_data.shape[-1]):

        for chn in channels_to_plot:
            plt.plot(eeg_data[0][:, chn], label=f'Channel :{chn}')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Filtering signals')
        plt.legend(ncol=10)
        plt.grid(True)
        plt.show()




# fig = plt.figure(figsize=(8, 6))
# fig.set_size_inches(11,11)
# ax = fig.add_subplot(111, projection="3d")
# fig.add_axes(ax)

# ax.set_title("EEG data")
# ax.view_init(azim=-30, elev=50)

# eeg_labels_np = np.array(eeg_labels)

# c_label = 30
# handles = []
# for chn_c in range(128):
 
#     # print(eeg_labels_np.shape)
#     filtered = np.where(eeg_labels_np == c_label)[0]
#     filtered_tsne = np.take(tsne_channel_wise[chn_c], filtered, 0)
#     # print(filtered_tsne.shape, chn_c)
#     _ = ax.text2D(0.8, 0.05, s=f"n_samples={filtered_tsne.shape[0]}", transform=ax.transAxes)
#     _patch = mpatches.Patch(color=channel_cmaps[chn_c], label=f'channel {chn_c}') 
#     handles.append(_patch)
#     ax.scatter(filtered_tsne[:,0], filtered_tsne[:,1],filtered_tsne[:,2], c=[channel_cmaps[chn_c] for i in range(filtered_tsne.shape[0])], s=50, alpha=0.8)
# plt.legend(handles=handles, loc="lower left", fontsize=10,ncol=4, bbox_to_anchor=(1.0, 0.0),fancybox=True) 