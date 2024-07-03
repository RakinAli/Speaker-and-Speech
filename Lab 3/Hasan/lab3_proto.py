import numpy as np
from lab3_tools import *
from lab3_tools import *

from lab1_tools import *
from lab1_proto import *

from lab2_tools import *
from lab2_proto import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """

    
    list =[]
    list.append('sil')
    for w in wordList:
       list.extend(pronDict[w])
       list.append("sp")

    list.append('sil') 
    return list
        

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans,state_Trans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])

    _, viterb_path = viterbi(obsloglik, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat'][:-1, :-1]))
    list = []
    for index in viterb_path:
        list.append(state_Trans[index])
    return list
