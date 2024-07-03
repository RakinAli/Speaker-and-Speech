import numpy as np
from lab2_tools import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

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
    # The number of emitting states in each HMM1 model
    hmm1_M = len(hmm1['startprob']) -1

    # π0 π1 π2 π3ρ0 π3ρ1 π3ρ2 π3ρ3, from document pdf
    startprob = np.concatenate((hmm1['startprob'][0:hmm1_M],hmm1['startprob'][-1] * hmm2['startprob']))

    hmm2_states = len(hmm2['startprob'])
    
    # Transition matrix for the concatenated model
    first_transition = np.hstack((hmm1['transmat'][0:hmm1_M,0:hmm1_M], np.outer(hmm1['transmat'][0:hmm1_M,-1], hmm2['startprob']) ))
    second_transition = np.hstack((np.zeros((hmm2_states, hmm1_M)), hmm2["transmat"]))
    transmat = np.vstack((first_transition, second_transition))

    # Means and covariances for the concatenated model
    means = np.vstack((hmm1['means'], hmm2['means']))
    covars = np.vstack((hmm1['covars'], hmm2['covars']))

    dict = {}
    dict['startprob'] = startprob
    dict['transmat'] = transmat
    dict['means'] = means
    dict['covars'] = covars

    return dict  


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

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
    for idx in range(1,len(namelist)):
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
        forward_prob: NxM array of forward log probabilities for each of the M states in the model:

    From the slides in the instructions:

        Log(a[0]i) = log(πi) + log(Bi(X[0]))

        Log(a[t]j) = log(Σi=1->N(a[t-1]i * a[i]j)) + log(Bj(X[t]))
    """

    forward_prob = np.zeros(log_emlik.shape)

    forward_prob[0,:] = log_startprob[:-1] + log_emlik[0]

    for n in range(1, forward_prob.shape[0]):
        for j in range(forward_prob.shape[1]):
            forward_prob[n,j] = logsumexp(forward_prob[n-1,:] + log_transmat[:-1,j]) + log_emlik[n,j]

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    
    Formula from the slides:
        Log (β[T-1][i]) = log(1) = 0
        Log (β[t][i]) = log(Σj=1->N(a[t][j] * b[j](X[t+1]) * β[t+1][j])) 

        Where a is the transition matrix, b is the emission matrix, and β is the backward matrix
    """

    backward_probs = np.zeros(log_emlik.shape)

    for timestep in reversed(range(backward_probs.shape[0]-1)):
        for i in range(backward_probs.shape[1]):
            backward_probs[timestep ,i] = logsumexp(log_transmat[i,:-1] + log_emlik[timestep+1,:] + backward_probs[timestep+1,:])

    return backward_probs


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

    # Initialize the first row of the viterbi log likelihood matrix
    for j in range(viterbi_loglik.shape[1]):
        viterbi_loglik[0][j] = log_startprob[j] + log_emlik[0][j]

    # Here we calculate the viterbi log likelihood matrix by taking the max of the previous row and adding the transition matrix
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

    # Backtracking and grabbing the best path
    for n in reversed(range(viterbi_backtrack.shape[0] - 1)):
        viterbi_path.append(int(viterbi_backtrack[n][viterbi_path[-1]]))
    viterbi_path.reverse()

    return np.max(viterbi_loglik[-1]), viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    
    Formula:
        log(gamma[n][i]) = log(alpha[n][i]) + log(beta[n][i]) - logsumexp(exp(log alpha[n-1][i] ))

    # Inuitively: 
        The gamma is the probability of being in state i at time n given the observation sequence X
    """
    log_gamma = np.zeros(log_alpha.shape)
    frame,state = log_alpha.shape

    for n in range(frame):
        for i in range(state):
            log_gamma[n,i] = log_alpha[n,i] + log_beta[n,i] - logsumexp(log_alpha[-1])

    return log_gamma


def updateMeanAndVar(x, log_gamma, variance_floor=5.0):
    """Update Gaussian parameters with diagonal covariance.

    Args:
         x (array): NxD array of feature vectors.
         log_gamma (array): NxM state posterior probabilities in log domain.
         variance_floor (float): minimum allowed variance scalar.

    Returns:
         tuple: means and covars arrays.
    """
    import numpy as np  # Ensure numpy is imported

    gamma = np.exp(log_gamma)  # Convert log probabilities to probabilities

    # Initialize means and covars
    means = np.zeros((log_gamma.shape[1], x.shape[1]))
    covars = np.zeros(means.shape)

    # Update means and covars for each state
    for k in range(means.shape[0]):
        means[k] = np.sum(gamma[:, k, np.newaxis] * x, axis=0) / np.sum(gamma[:, k])
        covars[k] = np.sum(
            gamma[:, k, np.newaxis] * (x - means[k]) ** 2, axis=0
        ) / np.sum(gamma[:, k])

    # Apply variance floor
    covars[covars < variance_floor] = variance_floor

    return means, covars
