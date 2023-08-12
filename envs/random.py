import numpy as np
import tqdm

from typing import Tuple


def random_mdp(
        horizon:int, n_states: int, n_actions: int, non_reachable_states: int = 0, zero_action: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Transition model
    if non_reachable_states > 0:
        transitions = np.random.random((horizon, n_states, n_actions, n_states))
        nr_states = np.random.choice(
            range(n_states), non_reachable_states, replace=False
        )
        eps = 1e-9
        for s in nr_states:
            transitions[:, :, s] = eps
    else:
        transitions = np.random.random((horizon, n_states, n_actions, n_states))


    reward = np.random.random((horizon, n_states, n_actions))
    reward -= 0.5
    #reward /= 2

    if zero_action:
        reward[:,:,0] = 0
        for state_source in range(n_states):
            for state_target in range(n_states):
                if state_target != state_source:
                    transitions[:, state_source, 0, state_target] = 0

    transitions /= transitions.sum(axis=3, keepdims=True)

    init_state_dist = np.random.random((n_states,))
    init_state_dist /= init_state_dist.sum()

    return transitions, reward, init_state_dist


