import numpy as np
import torch

def set_helpers_seed(seed):
    torch.manual_seed(seed)

from typing import Tuple


def random_argmax(array, axis=None):
    """Argmax with random tie breaking."""
    return np.argmax(
        np.random.random(array.shape)
        * (array == np.amax(array, axis=axis, keepdims=True)),
        axis=axis,
    )

def tie_breaker_argmax(array, array_tie_breaker, axis=None):
    """Argmax with given tie breaking."""
    amax = np.amax(array, axis=axis, keepdims=True)
    return np.argmax(
        array + np.nan_to_num(-np.inf) * (array != amax) + array_tie_breaker * (array == amax),
        axis=axis,
    )


def check_transitions_rewards(
    transitions: np.ndarray, rewards: np.ndarray
) -> Tuple[int, int]:
    assert len(rewards.shape) == 3, "Rewards need to have shape [H, S, A]"

    horizon, n_states, n_actions = rewards.shape

    assert transitions.shape == (
        horizon,
        n_states,
        n_actions,
        n_states,
    ), "Transitions need to have shape [H, S, A, S]"

    assert np.allclose(transitions.sum(axis=3), 1)

    return horizon, n_states, n_actions

def to_torch(lst):
    return [torch.from_numpy(np.asarray(x)).clone().float() if isinstance(x, (np.ndarray, list)) else x.clone().detach() for x in lst]

def ensure_policy_stochastic(policy, horizon, n_states, n_actions):
    if policy.shape == (horizon, n_states):
        policy = np.eye(n_actions)[policy]
    assert policy.shape == (horizon, n_states, n_actions)
    return policy

def sample_trajectory_from_visitation(d, mdp, init_state_dist):
    trajectory_states = []
    trajectory_actions = []

    horizon, n_states, n_actions, _ = mdp.shape

    if len(d.shape) == 1:
        d = d.reshape((horizon,n_states,n_actions))

    # Sample initial state
    current_state = torch.multinomial(init_state_dist, 1).item()
    trajectory_states.append(current_state)

    for h in range(horizon):
        # Get the state-action visitation measure for the current state at timestep h
        state_action_probs = d[h, current_state, :]

        # Sample the action based on the visitation measure
        action = torch.multinomial(state_action_probs, 1).item()
        trajectory_actions.append(action)

        # Get the transition probabilities for the current state and action
        transition_probs = mdp[h, current_state, action, :]

        # Sample the next state based on the transition probabilities
        next_state = torch.multinomial(transition_probs, 1).item()
        trajectory_states.append(next_state)

        # Update the current state
        current_state = next_state

    return trajectory_states, trajectory_actions

def sample_trajectory(mdp, pi, init_state_dist, horizon, n_states):
    trajectory_states = []
    trajectory_actions = []

    current_state = np.random.multinomial(init_state_dist, 1).item()
    trajectory_states.append(current_state)

    for h in range(horizon):
        action = pi[h, current_state].item()
        trajectory_actions.append(action)

        next_state = np.random.multinomial(mdp[h, current_state, action, :], 1).item()
        trajectory_states.append(next_state)

        current_state = next_state

    return trajectory_states, trajectory_actions

def vectorize_policy(
    pi: torch.Tensor,
    transitions: torch.Tensor,
    init_state_dist: torch.Tensor,
    horizon: int,
    n_states: int,
    n_actions: int,
):
    y = torch.zeros((horizon * n_states * n_actions))

    one_hot_actions = torch.eye(n_actions)

    y_prev_state_action_dist = (
        one_hot_actions[pi[0]] * init_state_dist[:, None]
    ).reshape(-1)
    y[: n_states * n_actions] = y_prev_state_action_dist

    transitions = transitions.reshape((horizon, n_states * n_actions, n_states))

    for h in range(1, horizon):
        y_current_state_dist = y_prev_state_action_dist @ transitions[h - 1]

        y_current_state_action_dist = (
            one_hot_actions[pi[h]] * y_current_state_dist[:, None]
        ).reshape(-1)

        y[
            h * n_states * n_actions : (h + 1) * n_states * n_actions
        ] = y_current_state_action_dist

        y_prev_state_action_dist = y_current_state_action_dist

    return y

def vectorize_uniform_policy(
    transitions: torch.Tensor,
    init_state_dist: torch.Tensor,
    horizon: int,
    n_states: int,
    n_actions: int,
    random_policy: bool = False
):
    y = torch.zeros((horizon * n_states * n_actions))
    
    if random_policy:
        # Sample a random policy
        random_policy_probs = torch.rand((n_states, n_actions))
        random_policy_probs /= random_policy_probs.sum(dim=1, keepdim=True)
    else:
        # For a uniformly random policy, the action distribution is uniform
        random_policy_probs = torch.ones((n_states, n_actions)) / n_actions
    
    # Initial state-action distribution
    y_prev_state_action_dist = (
        random_policy_probs.reshape(-1) * init_state_dist.repeat_interleave(n_actions)
    )
    y[: n_states * n_actions] = y_prev_state_action_dist
    
    transitions = transitions.reshape((horizon, n_states * n_actions, n_states))
    
    for h in range(1, horizon):
        # Compute next state distribution
        y_current_state_dist = y_prev_state_action_dist @ transitions[h - 1]
        
        # Compute next state-action distribution
        y_current_state_action_dist = (
            random_policy_probs.reshape(-1) * y_current_state_dist.repeat_interleave(n_actions)
        )
        
        y[h * n_states * n_actions : (h + 1) * n_states * n_actions] = y_current_state_action_dist
        y_prev_state_action_dist = y_current_state_action_dist
    
    return y
def fixed_n_rounding(allocation, N):
    """Rounding procedure to ensure the sum of allocation is N."""
    allocation_shape = allocation.shape
    allocation = allocation.reshape((-1,))
    allocation = np.ceil(allocation).astype(int)
    support = allocation > 1e-2
    where_support = np.where(support)[0]
    while np.sum(allocation) != N:
        if np.sum(allocation) > N:
            j = random_argmax(allocation[support])
            allocation[where_support[j]] -= 1
        else:  # np.sum(allocation) < N
            j = random_argmax(-allocation[support])
            allocation[where_support[j]] += 1
    allocation = allocation.reshape(allocation_shape)
    return allocation

def format_func(value, tick_number):
    exponent = int(np.log2(value))
    return f'$2^{{{exponent}}}$'
