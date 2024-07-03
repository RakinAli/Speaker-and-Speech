# DT2119, Lab 1 Feature Extraction

# Function given by the exercise 
# Hasans code
#----------------------------------
import numpy as np
import scipy as sc
from imports.lab1_tools import trfbank,lifter

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):

   
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    enframes = []
    start_index =0
    end_index = winlen


    
    while(end_index<=(len(samples))):
        temp =samples[start_index:end_index]
        enframes.append(temp.tolist())
        start_index +=winshift
        end_index += winshift
       
    
    return np.array(enframes) 


    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    """
    = x[n] + alpha* x[n-1]
    alpha = -p

    x[n] + alpha* x[n-1] does not exist of a0, therfore a =1
    Will be divition by 1
    """
    return np.array(sc.signal.lfilter(b=[1,-p],a=1,x=input))

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    N = input.shape[0]
    M = input.shape[1]

    """
    Make an window function that have the same size as 
    the input
    """
    
    return input * np.tile(sc.signal.windows.hamming(M,False),N).reshape((N,M))

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fft = sc.fftpack.fft(input,nfft)
    return np.square(np.abs(fft))

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    filters = trfbank(samplingrate, input.shape[1])
    return np.log((input @ filters.T))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    """
    [:, 0:nceps] the first coefficients 
    """
    return sc.fftpack.realtransforms.dct(x=input)[:,0:nceps] 

             # def dtw(x, y, dist):
def dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1)):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    N = len(x)
    M = len(y)
    local_distances = np.zeros((N, M))
    acc_dist = np.zeros((N, M))
    global_dist = 0
    path = []

    for i in range(N):
        for j in range(M):
            local_distances[i, j] = dist(x[i], y[j])

    rows, cols = local_distances.shape

    for row in range(1,rows):
        # Dynamic programming
        acc_dist[row, 0] = acc_dist[row - 1, 0] + local_distances[row, 0]
    
    # Fix row zero to handle the case when row is zero
    for col in range(1,cols):
        acc_dist[0, col] = acc_dist[0, col - 1] + local_distances[0, col]
    
    for row in range(1, rows):
        for col in range(1, cols):
            # Check for the minimum distance
            min_distance =  min(acc_dist[row - 1, col], acc_dist[row, col - 1], acc_dist[row - 1, col - 1])
            acc_dist[row, col] = min_distance + local_distances[row, col]

        
    global_dist = acc_dist[-1, -1]
    return global_dist