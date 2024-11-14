from typing import Tuple
import numpy as np

def driver_mdp(n_lanes: int = 5, length: int = 15, horizon: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_lanes % 2 == 0:
        raise ValueError("n_lanes must be odd")

    n_states = n_lanes * length
    n_actions = 3  # straight, left, right
    straight, left, right = 0, 1, 2

    # Initialize transitions
    transitions = np.zeros((horizon, n_states, n_actions, n_states))
    # Set up transitions for each state
    for state in range(n_states):
        lane, position = state // length, state % length
        
        # Straight action
        if position < length - 1:
            next_state = state + 1
        else:
            next_state = state  # Stay at the end
        transitions[:, state, straight, next_state] = 1.0
        
        # Left action (switch lane to left and move forward)
        if lane > 0 and position < length - 1:
            next_state = state - length + 1
        elif lane > 0 and position == length - 1:
            next_state = state - length  # Move left but stay at the end
        elif lane == 0 and position < length - 1:
            next_state = state + 1  # Move forward if in leftmost lane
        else:
            next_state = state  # Stay in place if at the end of leftmost lane
        transitions[:, state, left, next_state] = 1.0
        
        # Right action (switch lane to right and move forward)
        if lane < n_lanes - 1 and position < length - 1:
            next_state = state + length + 1
        elif lane < n_lanes - 1 and position == length - 1:
            next_state = state + length  # Move right but stay at the end
        elif lane == n_lanes - 1 and position < length - 1:
            next_state = state + 1  # Move forward if in rightmost lane
        else:
            next_state = state  # Stay in place if at the end of rightmost lane
        transitions[:, state, right, next_state] = 1.0        # Set up rewards (all zero as per requirement)
    reward = np.zeros(n_states)

    # Set up initial state distribution (start at the beginning of the middle lane)
    init_state_dist = np.zeros(n_states)
    middle_lane = n_lanes // 2
    init_state_dist[middle_lane * length] = 1.0

    # Create embedding phi
    phi = np.zeros((n_states, 2))
    for state in range(n_states):
        lane, position = state // length, state % length
        phi[state, 0] = position / (length - 1)  # Normalized position
        phi[state, 1] = (lane - n_lanes//2) / (n_lanes - 1)  # Absolute distance from middle lane

    return transitions, reward, init_state_dist, phi
