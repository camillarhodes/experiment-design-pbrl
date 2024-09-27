from typing import Tuple
import numpy as np

def river_swim_mdp(n_states: int = 6, horizon: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_actions = 2  # L and R actions
    left, right = 0, 1

    # Initialize transitions
    transitions = np.zeros((horizon, n_states, n_actions, n_states))

    # Set up transitions for each state
    for state in range(n_states):
        # Left action: always move left with probability 1
        l_state = max(0, state - 1)
        transitions[:, state, left, l_state] = 1.0

        # Right action: move right with 0.9 probability, left with 0.1 probability
        r_state = min(n_states - 1, state + 1)
        transitions[:, state, right, r_state] = 0.9
        transitions[:, state, right, l_state] = 0.1

    # Set up rewards (all zero as per requirement)
    reward = np.zeros(n_states)

    # Set up initial state distribution (start at leftmost state)
    init_state_dist = np.zeros(n_states)
    init_state_dist[0] = 1.0

    return transitions, reward, init_state_dist

def two_river_swim_mdp(n_states_per_river: int = 6, horizon: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_actions = 2  # L and R actions
    left, right = 0, 1
    total_states = 1 + 2 * n_states_per_river
    transitions = np.zeros((horizon, total_states, n_actions, total_states))
    
    # Starting state transitions
    transitions[:, 0, left, 1] = 1.0
    transitions[:, 0, right, n_states_per_river + 1] = 1.0
    
    for river in range(2):
        river_offset = river * n_states_per_river + 1
        for state in range(n_states_per_river):
            actual_state = river_offset + state
            
            # Left action
            if state == 0:  # Leftmost state of the river
                transitions[:, actual_state, left, actual_state] = 1.0  # Stay in the same state
            else:
                transitions[:, actual_state, left, actual_state - 1] = 1.0  # Move left
            
            # Right action
            if state == n_states_per_river - 1:  # Rightmost state of the river
                transitions[:, actual_state, right, actual_state] = 1.0  # Stay in the same state
            else:
                transitions[:, actual_state, right, actual_state + 1] = 0.9  # Move right with 0.9 probability
                transitions[:, actual_state, right, actual_state + 1] = 1
                transitions[:, actual_state, right, actual_state] = 0.1  # Stay with 0.1 probability
                transitions[:, actual_state, right, actual_state] = 0  # Stay with 0.1 probability

    reward = np.zeros(total_states)
    init_state_dist = np.zeros(total_states)
    init_state_dist[0] = 1.0
    return transitions, reward, init_state_dist

def modified_river_swim_mdp(n_actions: int = 5, n_states_per_river: int = 6, horizon: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rivers = 2  # Fixed to 2 rivers
    # Total number of states: 1 (starting state) + 2 * n_states_per_river
    total_states = 1 + 2 * n_states_per_river
    
    # Initialize transitions
    transitions = np.zeros((horizon, total_states, n_actions, total_states))
    
    # Set up transitions for the starting state (state 0)
    transitions[:, 0, 0, 1] = 1.0  # Action 0 moves to the start of the first river
    transitions[:, 0, 1, n_states_per_river + 1] = 1.0  # Action 1 moves to the start of the second river
    transitions[:, 0, 2:, 0] = 1.0  # All other actions stay in the starting state
    
    # Set up transitions for each river
    for river in range(2):
        river_offset = river * n_states_per_river + 1
        river_action = river  # Action 0 for first river, Action 1 for second river
        
        for state in range(n_states_per_river):
            actual_state = river_offset + state
            
            if state < n_states_per_river - 1:
                # The dedicated action moves right
                transitions[:, actual_state, river_action, actual_state + 1] = 1.0
                # All other actions stay in the same state
                for action in range(n_actions):
                    if action != river_action:
                        transitions[:, actual_state, action, actual_state] = 1.0
            else:
                # At the rightmost state, all actions stay in the same state
                transitions[:, actual_state, :, actual_state] = 1.0
    
    # Set up rewards (all set to 0 as before)
    reward = np.zeros(total_states)
    
    # Set up initial state distribution (start at the choosing state)
    init_state_dist = np.zeros(total_states)
    init_state_dist[0] = 1.0
    
    # Set up phi matrix
    phi = np.zeros((total_states, 2))
    
    # Calculate the normalization factor
    max_depth = n_states_per_river - 1
    
    # Phi for the first river: [normalized_depth, 0]
    for state in range(n_states_per_river):
        phi[state + 1, 0] = state / max_depth
    
    # Phi for the second river: [0, normalized_depth]
    for state in range(n_states_per_river):
        phi[n_states_per_river + state + 1, 1] = state / max_depth
    
    return transitions, reward, init_state_dist, phi
