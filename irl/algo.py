import numpy as np
import cvxpy as cp
import tqdm
import itertools

from typing import Tuple, Optional
from helpers import check_transitions_rewards


def get_X(horizon: int, n_states: int, n_actions: int) -> np.ndarray:
    states_eye = np.eye(n_states)
    horizon_eye = np.eye(horizon)
    vec = np.zeros(n_actions)
    vec[0] = 1
    X_zero_actions = np.kron(horizon_eye, np.kron(states_eye, vec))

    n = 1
    action_mat = np.zeros((n, n_actions))
    for action_plus in range(n_actions):
        for action_minus in range(action_plus + 1, n_actions):
            vec = np.zeros(n_actions)
            vec[action_plus] = 1
            vec[action_minus] = -1
            action_mat2 = np.zeros((n + 1, n_actions))
            action_mat2[:-1, :] = action_mat
            action_mat2[-1, :] = vec
            action_mat = action_mat2
            n += 1
    action_mat = action_mat[1:, :]

    X = np.kron(horizon_eye, np.kron(states_eye, action_mat))
    return X, X_zero_actions


def get_preference_probabilities(
    reward: np.ndarray, horizon: int, n_states: int, n_actions: int
) -> np.ndarray:
    assert (np.max(reward) <= 0.5) and (np.min(reward) >= -0.5)
    Y = None
    for h in range(horizon):
        for s in range(n_states):
            reward_action_pairs = np.array(
                list(itertools.combinations(reward[h, s, :], 2))
            )
            if h == 0 and s == 0:
                prob = 0.5 * (reward_action_pairs[:, 0] - reward_action_pairs[:, 1] + 1)
                # Y = 2*np.random.binomial(1, p)-1
            else:
                # Y = np.concatenate([Y, 2*np.random.binomial(1, p)-1])
                prob = np.concatenate(
                    [
                        prob,
                        0.5
                        * (reward_action_pairs[:, 0] - reward_action_pairs[:, 1] + 1),
                    ]
                )
                # Q[index_in_X] =  reward[h, s, action_plus] - reward[h, s, action_minus]
    return prob


def perform_uniform_allocation(
    X: np.ndarray,
    X_zero_actions: np.ndarray,
    reward: np.ndarray,
    horizon: int,
    n_states: int,
    n_actions: int,
    eps: float = 0.01,
    delta: float = 0.05,
    verbose=True,
) -> np.ndarray:
    Y = np.zeros(X.shape[0])

    step = len(Y)
    n_rounds = int(
        1 + 8 * n_states * n_actions * horizon**3 * np.log(2 / delta) / eps**2
    )
    if step < n_rounds:
        n_rounds += step - n_rounds % step
    if verbose:
        print(f"Performing uniform allocation with eps: {eps:.4f}, delta: {delta:.2f}")
        print(f"Number of rounds: {n_rounds}")

    prob = get_preference_probabilities(reward, horizon, n_states, n_actions)

    rng = range(0, n_rounds, step)
    it = tqdm.tqdm(rng) if verbose else range(rng)
    for r in it:
        Y_round = 2 * np.random.binomial(1, prob) - 1
        Y += Y_round
    Y /= n_rounds / step

    zero_actions = np.zeros(horizon * n_states)
    Y = np.concatenate([zero_actions, Y])
    X_combined = np.concatenate([X_zero_actions, X], axis=0)
    r_hat = np.linalg.inv(X_combined.T @ X_combined) @ X_combined.T @ Y
    return r_hat

def perform_RAGE(transitions: np.ndarray, r_hat: np.ndarray, init_state_dist: np.ndarray
        ) -> np.ndarray:
    opt = Optimizer()
    y1, b, A = opt._get_problem_params(transitions, r_hat, init_state_dist) 
    _ = opt._solve_opt_problem(y1, b, A)

    return None


class Optimizer:
    """Implements the convex optimization problem at the core of AceIRL."""

    def __init__(
        self,
        delta: float = 0.1,
    ):
        self.delta = delta

    def _solve_opt_problem(
            self,
            y1: np.ndarray,
            b: np.ndarray,
            A: np.ndarray
            ):

        # Objective
        #f = y1.T @ y1
        f = 1
        objective = cp.Minimize(f)

        # Constraints
        constraints = [A @ y1 == b, y1 >= 1e-24]

        #if use_eps_const:
        #    c_ = cp.vstack([c, np.ones((1, 1))])
        #    constraints.append(A1.T @ y_ >= c_)
        #else:
        #    constraints.append(A3.T @ y_ >= c)

        # Define the problem
        prob = cp.Problem(objective, constraints)

        # Solve thr problem 
        #eps = prob.solve(solver=cp.SCS, dqcp=True)
        obj_value = prob.solve()
        import ipdb; ipdb.set_trace()
        return None


    def _get_problem_params(
        self,
        P_hat: np.ndarray,
        R_hat: np.ndarray,
        init_state_dist: np.ndarray,
        #epsilon: float,
        #sample_count: np.ndarray,
        #n_ep_per_iter: int,
        init_state_dist_target: Optional[np.ndarray] = None,
        p_target: Optional[np.ndarray] = None,
        verbose: bool = False,
        next_step_n: bool = True,
    ):
        if p_target is None:
            p_target = P_hat
        if init_state_dist_target is None:
            init_state_dist_target = init_state_dist

        horizon, n_states, n_actions = check_transitions_rewards(P_hat, R_hat)
        assert p_target.shape == (horizon, n_states, n_actions, n_states)
        assert init_state_dist_target.shape == (n_states,)

        # Get optimal policy for estimated reward (in target)
        #_, V_hat, _ = value_iteration(p_target, R_hat)
        #V_hat = V_hat[0] @ init_state_dist_target

        # Variables
        mu_size = horizon * n_states * n_actions
        #y_size = horizon * n_states + horizon + 1
        mu = cp.Variable(mu_size, pos=True)
        #y = cp.Variable(y_size)

        ## b vector
        b1 = init_state_dist_target
        b2 = np.zeros(((horizon - 1) * n_states,))
        b3 = np.ones((horizon,))
        #b4 = -10 * epsilon * np.ones((1,))
        #b = np.concatenate([b1, b2, b3, b4]).reshape(-1)
        bb_ = np.concatenate([b1, b2, b3]).reshape(-1)

        #sample_count = sample_count.reshape((horizon * n_states * n_actions,))
        #sample_count = cp.maximum(sample_count, 1)

        #if next_step_n:
        #    n_tot = sample_count + mu * n_ep_per_iter
        #else:
        #    n_tot = sample_count

        #c = 2 * cp.multiply(
        #    (
        #        2
        #        * cp.log(
        #            24
        #            * n_states
        #            * n_actions
        #            * horizon
        #            * cp.square(sample_count)
        #            / self.delta
        #        )
        #    )
        #    ** 0.5,
        #    cp.power(n_tot, -0.5),
        #)

        #hh = np.arange(horizon).reshape((horizon, 1, 1))
        #hh = np.repeat(hh, n_states, axis=1)
        #hh = np.repeat(hh, n_actions, axis=2)
        #hh = np.reshape(hh, (horizon * n_states * n_actions,))
        #c = cp.multiply(c, horizon - hh)
        #c = cp.reshape(c, (horizon * n_states * n_actions, 1))

        # Matrix A
        Csi = np.kron(np.eye(n_states), np.ones((1, n_actions)))
        zero1 = np.zeros((n_states, n_states * n_actions))
        zero2 = np.zeros((n_states, 1))

        A10 = np.concatenate([Csi] + [zero1] * (horizon - 1) + [zero2], axis=1)
        #A_list1 = [A10]
        A_list2 = [A10[:, :-1]]
        for h in range(horizon - 1):
            P_target_ = p_target[h].reshape((n_states * n_actions, n_states))
            A_target = np.concatenate(
                [zero1] * h
                + [P_target_.T, -Csi]
                + [zero1] * (horizon - 2 - h)
                + [zero2],
                axis=1,
            )
            #A_list1.append(A_target)

            P_hat_ = P_hat[h].reshape((n_states * n_actions, n_states))
            A_hat = np.concatenate(
                [zero1] * h + [P_hat_.T, -Csi] + [zero1] * (horizon - 2 - h) + [zero2],
                axis=1,
            )
            A_list2.append(A_hat[:, :-1])
        A11 = np.concatenate(
            [
                np.kron(np.eye(horizon), np.ones((1, n_states * n_actions))),
                np.zeros((horizon, 1)),
            ],
            axis=1,
        )
        #A_list1.append(A11)
        A_list2.append(A11[:, :-1])
        #A12 = np.concatenate(
        #    [
        #        np.concatenate(
        #            R_hat.reshape((horizon, n_states * n_actions)), axis=0
        #        ).reshape((1, -1)),
        #        -np.ones((1, 1)),
        #    ],
        #    axis=1,
        #)
        #A_list1.append(A12)

        #A1 = np.concatenate(A_list1, axis=0)
        A2 = np.concatenate(A_list2, axis=0)
        #A3 = np.concatenate(A_list1[:-1], axis=0)
        #A4 = A_list1[-1]

        #return mu, y, b, c, bb_, A1, A2, A3, A4
        return mu, bb_, A2


