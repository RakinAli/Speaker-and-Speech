import numpy as np

from lab3_tools import *
from lab3_proto import *

from imports.lab1_tools import *
from imports.lab1_proto import *
from imports.lab2_tools import *
from imports.lab2_proto import *


from imports.prondict import prondict

from imports.lab1_proto import mfcc

import os
from tqdm import tqdm
path = 'Data/'

def main():
    # Grab the HMMs from the lab2_models_all.npz file
    phoneHMMs = np.load("../../Lab 2/Rakin/lab2_models_all.npz", allow_pickle=True)[
        "phoneHMMs"
    ].item()

    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]["means"].shape[0] for phone in phones}
    stateList = [ph + "_" + str(id) for ph in phones for id in range(nstates[ph])]
    print("State List: ", stateList)

    # From the instructions, to be changed later
    filename = "tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav"
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)

    wordTrans = list(path2info(filename)[2])
    print("wordTrans: ", wordTrans)
    phoneTrans = words2phones(wordTrans, prondict)
    print("phoneTrans: ", phoneTrans)

    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    stateTrans = [
        phone + "_" + str(stateid)
        for phone in phoneTrans
        for stateid in range(nstates[phone])
    ]
    print("stateTrans, ", stateTrans)

    obsloglik = log_multivariate_normal_density_diag(
        lmfcc, utteranceHMM["means"], utteranceHMM["covars"]
    )
    path = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
    frames2trans(path, outfilename="z43a.lab")  


    # Check if there are files inside "Data" directory. Is there are then skip this step
    if not os.path.exists("Data"):
        os.makedirs("Data")
        # First, calculate the total number of .wav files
        total_files = 0
        # Make sure to switch this to either test or train
        root_dir = "tidigits/disc_4.2.1/tidigits/test"
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    total_files += 1

        # Create a tqdm progress bar with the total count of files
        train_data = []
        pbar = tqdm(total=total_files, desc="Processing files")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    filename = os.path.join(root, file)
                    samples, samplingrate = loadAudio(filename)
                    lmfcc = mfcc(samples)
                    mspect_data = mspec(samples)
                    wordTrans = list(path2info(filename)[2])
                    phoneTrans = words2phones(wordTrans, prondict)
                    path = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
                    train_data.append(
                        {
                            "filename": filename,
                            "lmfcc": lmfcc,
                            "mspect": mspect_data,
                            "targets": path,
                        }
                    )
                    pbar.update(1)  # Update the progress bar for each file processed
        pbar.close()

        # Save the training data to a compressed NumPy file
        np.savez("test.npz", train_data=train_data)
    else:
        print("Data directory already exists. Skipping data converting stage.")
        print("Starting data pre-processing...")
    
    # Load the training dataset from data
    data = np.load("Data/train_data.npz", allow_pickle=True)["train_data"]
    print("Data loaded successfully.")

    # Split the training set 90 train 10 validation. Make sure there is similar distribution of women in both sets that that each speaker is only included in one of thw two set

    split = 0.9
    n = len(data)
    np.random.seed(0)
    indices = np.random.permutation(n)
    ntrain = int(n * split)
    train_data = [data[i] for i in indices[:ntrain]]
    valid_data = [data[i] for i in indices[ntrain:]]
    print("Data split successfully.")

    

    

    


if __name__ == "__main__":
    main()
