import numpy as np
from typing import Tuple

def alternating_path_mdp(n_states=10, horizon=20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_actions = 2
    
    # Transition dynamics
    transitions = np.zeros((horizon, n_states, n_actions, n_states))
    for state in range(n_states - 1):
        transitions[:, state, :, state + 1] = 1  # Both actions lead to next state
    transitions[:, n_states - 1, :, n_states - 1] = 1  # Stay in last state
    
    # Reward structure
    reward = np.zeros((horizon, n_states, n_actions))
    reward[:, :, 0] = -0.5  # First action always gives -0.5 reward
    reward[:, :, 1] = 0.5   # Second action always gives 0.5 reward
    
    # Initial state distribution
    init_state_dist = np.zeros(n_states)
    init_state_dist[0] = 1  # Always start in state 0
    
    return transitions, reward, init_state_dist
