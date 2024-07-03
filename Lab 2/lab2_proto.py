import numpy as np
from lab2_tools import *


# Hasans code
def concatTwoHMMs(hmm1, hmm2):
    """Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    hmm1_M = (
        len(hmm1["startprob"]) - 1
    )  # the number of emitting states in each HMM1 model

    # π0 π1 π2 π3ρ0 π3ρ1 π3ρ2 π3ρ3, from document pdf
    startprob = np.concatenate(
        (hmm1["startprob"][0:hmm1_M], hmm1["startprob"][-1] * hmm2["startprob"])
    )

    hmm2_start = len(hmm2["startprob"])

    first_part_transition = np.hstack(
        (
            hmm1["transmat"][0:hmm1_M, 0:hmm1_M],
            np.outer(hmm1["transmat"][0:hmm1_M, -1], hmm2["startprob"]),
        )
    )
    second_part_transition = np.hstack(
        (np.zeros((hmm2_start, hmm1_M)), hmm2["transmat"])
    )
    transmat = np.vstack((first_part_transition, second_part_transition))

    means = np.vstack((hmm1["means"], hmm2["means"]))
    covars = np.vstack((hmm1["covars"], hmm2["covars"]))

    dict = {}
    dict["startprob"] = startprob
    dict["transmat"] = transmat
    dict["means"] = means
    dict["covars"] = covars

    return dict


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name.
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

    # Page 7 in the book
    # recursion formula in the lab instruction, NxM
    forward_prob = np.zeros((log_emlik.shape))

    # Initialization step
    for j in range(forward_prob.shape[1]):
        forward_prob[0][j] = log_startprob[j] + log_emlik[0][j]

    # Recursion step, t =>observations, s  => states
    for t in range(1, forward_prob.shape[0]):
        for s in range(forward_prob.shape[1]):
            forward_prob[t][s] = (
                logsumexp(forward_prob[t - 1] + log_transmat[:, s]) + log_emlik[t][s]
            )

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

    """
    Implmentation with help of formulas in  appendiz
    No need to intilaization step since it zero, and we use array that contain zeros in the begining 
    """

    backward_prob = np.zeros((log_emlik.shape))

    for t in range((backward_prob.shape[0]) - 2, -1, -1):
        for s in range(backward_prob.shape[1]):
            backward_prob[t][s] = logsumexp(
                log_transmat[s, :] + log_emlik[t + 1, :] + backward_prob[t + 1, :]
            )

    return backward_prob


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

    viterbi_loglik = np.zeros(log_emlik.shape)

    viterbi_backtrack = np.zeros(log_emlik.shape)

    for j in range(viterbi_loglik.shape[1]):
        viterbi_loglik[0][j] = log_startprob[j] + log_emlik[0][j]

    for t in range(1, viterbi_loglik.shape[0]):
        for s in range(viterbi_loglik.shape[1]):
            viterbi_loglik[t, s] = (
                np.max(viterbi_loglik[t - 1, :] + log_transmat[:, s]) + log_emlik[t, s]
            )
            viterbi_backtrack[t, s] = np.argmax(
                viterbi_loglik[t - 1, :] + log_transmat[:, s]
            )

    viterbi_path = []
    viterbi_path.append(np.argmax(viterbi_loglik[-1]))

    for n in reversed(range(viterbi_backtrack.shape[0] - 1)):
        viterbi_path.append(int(viterbi_backtrack[n][viterbi_path[-1]]))
    viterbi_path.reverse()
    print("Viterbi working")

    return np.max(viterbi_loglik[-1]), viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    log_gamma = np.zeros((log_alpha.shape))
    for t in range(log_gamma.shape[0]):
        for s in range(log_gamma.shape[1]):
            log_gamma[t][s] = (
                log_alpha[t][s] + log_beta[t][s] - logsumexp(log_alpha[-1, :])
            )
            # Verify that the sum of the gamma probabilities at each time frame is 1
            assert np.isclose(np.sum(np.exp(log_gamma[t])), 1.0)

    return log_gamma


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
