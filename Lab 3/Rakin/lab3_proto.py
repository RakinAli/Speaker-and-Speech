import numpy as np
from lab3_tools import *
from imports.lab2_tools import *
from imports.lab2_proto import *
from imports.prondict import prondict


def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phones = []
    for word in wordList:
        phones += pronDict[word]
        if addShortPause:
            phones += ["sp"]
    if addSilence:
        phones = ["sil"] + phones + ["sil"]
    return phones


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step the state from phoneHMMs corresponding to the viterbi path.
    """

    # Basically do Viterbi path then translate it to state names and return it

    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]["means"].shape[0] for phone in phones}

    HMM_concated = concatHMMs(phoneHMMs, phoneTrans)
    stateTrans = [
        phone + "_" + str(stateid)
        for phone in phoneTrans
        for stateid in range(nstates[phone])
    ]

    obsloglik = log_multivariate_normal_density_diag(
        lmfcc, HMM_concated["means"], HMM_concated["covars"]
    )

    viterbiStateTrans = viterbi(
        obsloglik,
        np.log(HMM_concated["startprob"] + np.finfo("float").eps),
        np.log(HMM_concated["transmat"][:-1, :-1] + np.finfo("float").eps),
    )[1]

    aligned_state_names = [stateTrans[state] for state in viterbiStateTrans]

    return aligned_state_names
