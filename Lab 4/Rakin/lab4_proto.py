import torch
import torchaudio
# DT2119, Lab 4 End-to-end Speech Recognition

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(n_mels=80,sample_rate=16000), torchaudio.transforms.FrequencyMasking(freq_mask_param=15), torchaudio.transforms.TimeMasking(time_mask_param=35))
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = torchaudio.transforms.MelSpectrogram(n_mels=80,sample_rate=16000)

# Functions to be implemented ----------------------------------


def strToInt(text):
    """
        Convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    """

    characters = [
        "’",
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    labels = []

    for char in text:
        try:
            labels.append(characters.index(char))
        except ValueError:
            print(f"Character '{char}' not found in character list.")

    return labels


def intToStr(labels):
    """
        Convert list of integers to string
    Args:
        labels: list of ints
    Returns:
        string with space-separated characters
    """

    characters = [
        "’",
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    word = ""
    for label in labels:
        label = int(label)  # Ensure the label is an integer
        if 0 <= label < len(characters):
            word += characters[label]
        else:
            print(f"Label '{label}' is out of range.")
    return word



def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    all_spec = []
    all_labels= []
    input_lengths = []
    label_lengths = []
    for i in range(len(data)): 
        waveform = data[i][0]
        utterance= data[i][2]

        labels =  torch.tensor(strToInt(utterance.lower()),dtype=torch.float32)
        label_lengths.append(len(labels))
        all_labels.append(labels)
        
        spec = transform(waveform) # Channel, n_mels, time 
        spec = spec.squeeze(0).transpose(0, 1) # time, n_mels
        input_lengths.append(int(spec.shape[0]/2))
        all_spec.append(spec)

    X = torch.nn.utils.rnn.pad_sequence(all_spec, batch_first=True) 
    X = X.unsqueeze(1).transpose(2, 3)
    
    return X, torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True) ,  input_lengths, label_lengths

def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    
    decoded = []
    for i in range(output.shape[0]):
        sequence = output[i]
        sequence = torch.argmax(sequence, dim=1)
        sequence = sequence.tolist()
        sequence = [x for x in sequence if x != blank_label]
        decoded.append(intToStr(sequence))

    return decoded


def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence - seq 1
        hyp: sequence to compare against the reference - seq 2
    output:
        edit distance (int)
    '''
    # Initialize matrix
    matrix = [[0 for x in range(len(hyp)+1)] for y in range(len(ref)+1)]

    # Fill the first row of matrix with values ranging from 0 to len(hyp)
    for i in range(len(hyp)+1):
        matrix[0][i] = i

    # Fill the first column of matrix with values ranging from 0 to len(ref)
    for i in range(len(ref)+1):
        matrix[i][0] = i

    # Fill the rest of the matrix
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1]) + 1

    return matrix[len(ref)][len(hyp)]
