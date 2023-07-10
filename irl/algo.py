import numpy as np
import tqdm
import itertools

from typing import Tuple



def get_design_matrix(horizon: int, n_states: int, n_actions: int, zero_action: int = 1) -> np.ndarray:
    X = get_X_matrix(horizon, n_states, n_actions, zero_action)
    return X.T@X

def get_X(horizon: int, n_states: int, n_actions: int, zero_action: int = 1) -> np.ndarray:

    states_eye = np.eye(n_states)
    horizon_eye = np.eye(horizon)
    vec = np.zeros(n_actions)
    vec[0] = 1
    X_zero_actions =  np.kron(horizon_eye, np.kron(states_eye, vec))


    n = 1
    action_mat = np.zeros((n, n_actions))
    for action_plus in range(n_actions):
        for action_minus in range(action_plus+1, n_actions):
            vec = np.zeros(n_actions)
            vec[action_plus] = 1
            vec[action_minus] = -1
            action_mat2 = np.zeros((n+1, n_actions))
            action_mat2[:-1,:] = action_mat
            action_mat2[-1,:] = vec
            action_mat = action_mat2
            n += 1
    action_mat = action_mat[1:,:]

    X =  np.kron(horizon_eye, np.kron(states_eye, action_mat))
    return X, X_zero_actions

def get_preference_probabilities(X: np.ndarray, reward: np.ndarray, horizon: int, n_states: int, n_actions: int, zero_action: int = 1, fast=True) -> np.ndarray:

    assert (np.max(reward) <= 0.5) and (np.min(reward) >= -0.5)
    Y = None
    for h in range(horizon):
        for s in range(n_states):
            reward_action_pairs = np.array(list(itertools.combinations(reward[h, s, :], 2)))
            if h==0 and s==0:
                prob = 0.5*(reward_action_pairs[:,0]-reward_action_pairs[:,1] + 1)
                    #Y = 2*np.random.binomial(1, p)-1
            else:
                    #Y = np.concatenate([Y, 2*np.random.binomial(1, p)-1])
                prob = np.concatenate([prob, 0.5*(reward_action_pairs[:,0]-reward_action_pairs[:,1] + 1)])
                    #Q[index_in_X] =  reward[h, s, action_plus] - reward[h, s, action_minus]
    return prob

def perform_uniform_allocation(X: np.ndarray, X_zero_actions: np.ndarray, reward: np.ndarray, horizon: int,  n_states: int, n_actions: int, zero_action: int = 1, eps: float=0.01, delta: float=0.05, verbose=True) -> np.ndarray:

    Y = np.zeros(X.shape[0])
    step = len(Y)
    n_rounds = int(1+8*n_states*n_actions*horizon**3*np.log(2/delta) / eps**2)
    if step < n_rounds:
        n_rounds += step - n_rounds % step
    if verbose:
        print(f'Performing uniform allocation with eps: {eps:.4f}, delta: {delta:.2f}')
        print(f'Number of rounds: {n_rounds}')


    prob = get_preference_probabilities(X, reward, horizon, n_states, n_actions, zero_action)

    #prob = np.tile(prob, step)
    rng = range(0, n_rounds, step)
    it = tqdm.tqdm(rng) if verbose else range(rng)
    for r in it:
        Y_round = 2*np.random.binomial(1, prob)-1
        Y += Y_round
    Y /= (n_rounds/step)

    zero_actions = np.zeros(horizon*n_states)
    Y = np.concatenate([zero_actions, Y])
    X_combined = np.concatenate([X_zero_actions, X], axis=0)
    r_hat = np.linalg.inv(X_combined.T@X_combined)@X_combined.T@Y
    return r_hat


