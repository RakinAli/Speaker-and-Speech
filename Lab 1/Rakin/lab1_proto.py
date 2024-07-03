# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
# Rakin
import numpy as np
import scipy as sc
from lab1_tools import trfbank, lifter
import matplotlib.pyplot as plt
import cdist as euclidean
from scipy.fftpack import dct

import sklearn.mixture as mix

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
        temp = samples[start_index:end_index]
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
    Will be division by 1
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
    fourier = np.fft.fft(input, nfft)
    answer = np.square(np.abs(fourier))
    return answer

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
    real_transforms = dct(x=input, norm="ortho")[:, 0:nceps]
    return real_transforms

def get_mspec(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


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
    


def part_one():
    # Create a 5 row 1 column array subplot grid for all the plots
    fig, axs = plt.subplots(8, 1, figsize=(20, 20))  # Set the size of the figure

    # Adjust the spacing between subplots
    plt.subplots_adjust(
        hspace=1.5, top=0.95, bottom=0.05
    )  # Increase the height between subplots
    plt.title("Example data")

    # Plot the sample
    example = np.load("lab1_example.npz", allow_pickle=True)["example"].item()[
        "samples"
    ]
    axs[0].plot(example)
    axs[0].set_title("Samples")
    axs[0].set_yticks([])

    # Enframe
    example = np.load("lab1_example.npz", allow_pickle=True)["example"].item()
    samples = example["samples"]
    enframed = enframe(samples, 400, 200)
    axs[1].pcolormesh(enframed.T)
    axs[1].set_title("Enframed")

    # Pre-emphasis
    preemph = preemp(enframed)
    axs[2].pcolormesh(preemph.T)
    axs[2].set_title("Pre-emphasis")

    # Windowing
    windowed = windowing(preemph)
    axs[3].pcolormesh(windowed.T)
    axs[3].set_title("Windowing")

    # Power spectrum
    spec = powerSpectrum(windowed, 512)
    axs[4].pcolormesh(spec.T)
    axs[4].set_title("Power Spectrum")

    # Mel spectrum
    mel = logMelSpectrum(spec, 20000)
    axs[5].pcolormesh(mel.T)
    axs[5].set_title("Mel Spectrum")

    # Cepstrum
    ceps = cepstrum(mel, 13)
    axs[6].pcolormesh(ceps.T)
    axs[6].set_title("Cepstrum")

    # lifter
    liftered = lifter(ceps, 22)
    axs[7].pcolormesh(liftered.T)
    axs[7].set_title("Liftered")

    plt.show()


def plot_gmm_for_utterance(utterance, components, data, full_data=None):
    fig, axs = plt.subplots(
        1, len(components), figsize=(20, 4)
    )  # 1 row, N columns for components

    for i, component in enumerate(components):
        mfcc_sample = mfcc(
            data[utterance]["samples"]
        )  # Replace with your actual data retrieval
        model = mix.GaussianMixture(n_components=component)
        # Fit the matrix data
        if full_data is not None:
            model.fit(full_data)
        else:
            # Fit the sample data
            model.fit(mfcc_sample)
        posterior = model.predict_proba(mfcc_sample)

        # Normalize the posterior probabilities
        norm_posterior = posterior / posterior.sum(axis=0)

        # Plotting the normalized posterior for each component in a subplot
        cax = axs[i].pcolormesh(norm_posterior, shading="auto")
        axs[i].set_title(f"Utterance {utterance}: {component} components", pad=20)
        axs[i].set_xlabel("Samples")
        if i == 0:  # Only set the y-axis label for the first subplot
            axs[i].set_ylabel("Components")

        # Adding a colorbar to the last subplot
        if i == len(components) - 1:
            fig.colorbar(cax, ax=axs[i], orientation="vertical")

    # Adjust layout
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=1.0)
    plt.show()


def main():
    """
    print("\033[91;1m_____Testing example_____\033[0m")
    part_one()
    print("\033[91;1m_____Testing with data _____\033[0m")
    """
    # Part 2
    print("\033[91;1m_____Part 5_____\033[0m")
    data = np.load('lab1_data.npz', allow_pickle=True)['data']

    mfcc_data = []
    mspec_data = []

    for audio in data:
        # Get the mfcc and mspec data
        mfcc_sample = mfcc(audio["samples"])
        mel_filter_sample = get_mspec(audio["samples"])

        # Append to the lists
        mfcc_data.append(mfcc_sample)
        mspec_data.append(mel_filter_sample)

    mfcc_matrix = np.vstack(mfcc_data)
    mspec_matrix = np.vstack(mspec_data)

    
    mfcc_coeff = np.corrcoef(mfcc_matrix, rowvar=False)
    mspec_coeff = np.corrcoef(mspec_matrix, rowvar=False)

    plt.pcolormesh(mfcc_coeff)
    plt.colorbar()
    plt.title("MFCC correlation")
    plt.show()

    plt.pcolormesh(mspec_coeff)
    plt.colorbar()
    plt.title("MSPEC correlation")
    plt.show()
    
    print("\033[91;1m_____Part 6_____\033[0m")

    components = [4, 8, 16, 32]
    utterances = [16,17,38,39]


    # Iterate over utterances and components and plot
    for utterance in utterances:
        plot_gmm_for_utterance(utterance, components, data,full_data=mfcc_matrix)
    
    print("\033[91;1m_____Part 7_____\033[0m")


if __name__ == "__main__":
    main()
