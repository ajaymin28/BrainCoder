from utils.eeg_utils import EEG_Dataset2

class args:
    dnn = "clip"

for Subi in range(1,11):
    ds = EEG_Dataset2(args=args,nsub=Subi,agument_data=False,
                    load_individual_files=False,
                    subset="train",
                    preTraning=True,
                    include_neg_sample=False,
                    saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")

    ds.save_ds_files(save_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")