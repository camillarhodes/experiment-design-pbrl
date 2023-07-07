import numpy as np
import tqdm

from typing import Tuple


def random_mdp(
        n_states: int, n_actions: int, horizon: int, non_reachable_states: int = 0, zero_action: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Transition model
    if non_reachable_states > 0:
        transitions = np.random.random((horizon, n_states, n_actions+zero_action, n_states))
        nr_states = np.random.choice(
            range(n_states), non_reachable_states, replace=False
        )
        eps = 1e-9
        for s in nr_states:
            transitions[:, :, s] = eps
    else:
        transitions = np.random.random((horizon, n_states, n_actions+zero_action, n_states))


    reward = np.random.random((horizon, n_states, n_actions))
    reward -= 0.5
    #reward /= 2

    if zero_action:
        reward[:,:,0] = 0
        for state_source in range(n_states):
            for state_target in range(n_states):
                if state_target != state_source:
                    transitions[:, state_source, -1, state_target] = 0

    transitions /= transitions.sum(axis=3, keepdims=True)

    init_state_dist = np.random.random((n_states,))
    init_state_dist /= init_state_dist.sum()

    return transitions, reward, init_state_dist

def get_design_matrix(n_states: int, n_actions: int, horizon: int, zero_action: int = 1) -> np.ndarray:
    X = get_X_matrix(n_states, n_actions, horizon, zero_action)
    return X.T@X

def get_X_matrix(n_states: int, n_actions: int, horizon: int, zero_action: int = 1, rounds: int = 1) -> np.ndarray:

    n = 1
    mat = np.zeros((n, n_actions))
    if zero_action:
        mat[0,0] = 1
    for action_plus in range(n_actions):
        for action_minus in range(action_plus+1, n_actions):
            vec = np.zeros(n_actions)
            vec[action_plus] = 1
            vec[action_minus] = -1
            mat2 = np.zeros((n+1, n_actions))
            mat2[:-1,:] = mat
            mat2[-1,:] = vec
            mat = mat2
            n += 1
    states_eye = np.eye(n_states)
    horizon_eye = np.eye(horizon)
    return np.kron(horizon_eye, np.kron(states_eye, mat))

def get_action_preferences(X: np.ndarray, reward: np.ndarray, n_states: int, n_actions: int, horizon: int, zero_action: int = 1) -> np.ndarray:

    assert (np.max(reward) <= 0.5) and (np.min(reward) >= -0.5)
    Y = np.zeros(X.shape[0])
    Q = np.zeros(X.shape[0])
    for h in range(horizon):
        for s in range(n_states):
            for action_plus in range(n_actions):
                for action_minus in range(action_plus+1, n_actions):
                    index_in_X = int((h*n_states+s)*(zero_action+n_actions*(n_actions-1)/2) \
                            + action_plus + action_minus -1 + zero_action)
                    vec = np.zeros(X.shape[1])
                    vec[int((h*n_states+s)*(n_actions*(n_actions-1)/2))+action_plus] = 1
                    vec[int((h*n_states+s)*(n_actions*(n_actions-1)/2))+action_minus] = -1
                    if not all(X[index_in_X] == vec):
                        import pdb; pdb.set_trace()
                    p = 0.5*(reward[h, s, action_plus] - reward[h, s, action_minus] + 1)
                    Y[index_in_X] =  2*np.random.binomial(1, p, 1)-1
                    #Q[index_in_X] =  reward[h, s, action_plus] - reward[h, s, action_minus]
    return Y

def perform_uniform_allocation(X: np.ndarray, reward: np.ndarray, n_states: int, n_actions: int, horizon: int, zero_action: int = 1, n_rounds: int=1, verbose=True) -> np.ndarray:
    Y = None
    print(verbose)
    it = tqdm.tqdm(range(n_rounds)) if verbose else range(n_rounds)
    for r in it:
        Y_round = get_action_preferences(X, reward, n_states, n_actions, horizon, zero_action)
        if Y is None:
            Y = Y_round
        else:
            Y = np.concatenate([Y, Y_round])
    return np.tile(X, (n_rounds, 1)), Y


