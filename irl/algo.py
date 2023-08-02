import numpy as np
import cvxpy as cp
import tqdm
import itertools

from typing import Tuple, Optional
from helpers import check_transitions_rewards
from rl.value_iteration import value_iteration, policy_evaluation


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
    it = tqdm.tqdm(rng) if verbose else rng
    for r in it:
        Y_round = 2 * np.random.binomial(1, prob) - 1
        Y += Y_round
    Y /= n_rounds / step

    zero_actions = np.zeros(horizon * n_states)
    Y = np.concatenate([zero_actions, Y])
    X_combined = np.concatenate([X_zero_actions, X], axis=0)
    r_hat = np.linalg.inv(X_combined.T @ X_combined) @ X_combined.T @ Y
    return r_hat


def perform_RAGE(
        X: np.ndarray,
        X_zero_actions: np.ndarray,
        reward: np.ndarray,
        horizon: int,
        n_states: int,
        n_actions: int,
        #reward_hat: np.ndarray,
        transitions: np.ndarray,
        init_state_dist: np.ndarray,
        verbose: bool = True,
        ) -> np.ndarray:

    np.random.seed(42)

    rage_max_iter = 30
    X_combined = np.concatenate([X_zero_actions, X], axis=0)
    rng = range(1, rage_max_iter)
    it = tqdm.tqdm(rng) if verbose else rng
    for rage_iter in it:
        r_hat = np.random.rand(*reward.shape)

        _, V_hat, _ = value_iteration(transitions, r_hat)
        V_hat_opt = init_state_dist @ V_hat[0]
        V_hat_target = V_hat_opt - 2**(-(rage_iter+2))

        design = get_optimal_design(X_combined, r_hat, V_hat_target, horizon, n_states, n_actions, transitions, init_state_dist, verbose)

    return None

def get_optimal_design(
        X_combined: np.ndarray,
        r_hat: np.ndarray,
        V_hat_target: float,
        horizon: int,
        n_states: int,
        n_actions: int,
        transitions: np.ndarray,
        init_state_dist: np.ndarray,
        verbose: bool = True,
        ) -> np.ndarray:
    
    design = np.ones(len(X_combined))
    design /= design.sum()

    max_iter = 1000
    batch_size = 20

    rng = range(1, max_iter)
    it = tqdm.tqdm(rng) if verbose else rng

    for iter in it:
        design_inv = np.linalg.inv(X_combined.T@ np.diag(design) @ X_combined)
        # compute gradient with respect to lambda and solve linear problem
        g_indices = []
        for i in range(batch_size):
            y1, y2 = get_opt_y1_y2(transitions, init_state_dist, r_hat, V_hat_target, design_inv, horizon, n_states, n_actions)
            #import ipdb;  ipdb.set_trace()
            g = ((X_combined @ design_inv @ (y1-y2)) * (X_combined @ design_inv @ (y1-y2))).flatten()
            g_idx = np.argmax(g)
            g_indices.append(g_idx)

        if verbose and iter % 200 == 1 and False:
            true_uncertainty = get_true_uncertainty(transitions, init_state_dist, design_inv, horizon, n_states, n_actions)
            print(f'Current true uncertainty: {true_uncertainty}')
            print(f'Current estimated uncertainty: {(y1-y2).T@design_inv@(y1-y2)}')

        # perform frank-wolfe update with fixed stepsize
        gamma = (2 / (iter + 2))/len(g_indices)
        relative_sum = 0
        for g_idx in g_indices:
            design_update = -gamma * design
            design_update[g_idx] += gamma
            design += design_update
            relative_sum += np.linalg.norm(design_update) / (np.linalg.norm(design)) 


        
        relative = relative_sum/batch_size

        if relative < 0.001:  # stop if change in last step is small
            print(f'Frank Wolfe detected relative < 0.001, aborting')
            if verbose and False:
                true_uncertainty = get_true_uncertainty(transitions, init_state_dist, design_inv, horizon, n_states, n_actions)
                print(f'Current true uncertainty: {true_uncertainty}')
            break

    print(f'Frank Wolfe finished after iter {iter}')
    idx_fix = np.where(design < 1e-5)[0]
    drop_total = design[idx_fix].sum()
    design[idx_fix] = 0
    design[np.argmax(design)] += drop_total
    return design

def get_opt_y1_y2(
        transitions: np.ndarray,
        init_state_dist: np.ndarray,
        r_hat: np.ndarray,
        V_hat_target: float,
        design_inv: np.ndarray,
        horizon: int,
        n_states: int,
        n_actions: int,
        ):

    def get_constrained_policy(rew):
        left = 0
        right = 0
        tol = 0.01
        pi = value_iteration(transitions, rew+right*r_hat)[2]
        while policy_evaluation(transitions, r_hat, pi)[1][0] @ init_state_dist < V_hat_target:
            if right == 0:
                right = 1
            else:
                right *= 2
            pi = value_iteration(transitions, rew+right*r_hat)[2]

        while right - left > tol:
            mid = (right + left)/2
            pi = value_iteration(transitions, rew+mid*r_hat)[2]
            V_hat_current = policy_evaluation(transitions, r_hat, pi)[1][0] @ init_state_dist
            if V_hat_current < V_hat_target:
                left = mid
            else:
                right = mid
        
        if right != 0:
            pi =  value_iteration(transitions, rew+right*r_hat)[2]
        assert policy_evaluation(transitions, r_hat, pi)[1][0] @ init_state_dist >= V_hat_target
        return pi



    
    eigenvalues, eigenvectors=np.linalg.eigh(design_inv)
    v = eigenvectors[np.random.choice(np.arange(len(eigenvalues)), p=eigenvalues/np.sum(eigenvalues))]
    v = v.reshape((horizon, n_states, n_actions))
    pi_y1 = get_constrained_policy(v)
    pi_y2 = get_constrained_policy(-v)

    y1 = vectorize_policy(pi_y1, transitions, init_state_dist, horizon, n_states, n_actions)
    y2 = vectorize_policy(pi_y2, transitions, init_state_dist, horizon, n_states, n_actions)

    return y1, y2

def get_true_uncertainty(
        transitions: np.ndarray,
        init_state_dist: np.ndarray,
        design_inv: np.ndarray,
        horizon: int,
        n_states: int,
        n_actions: int,

        ):

    import string
    digs = string.digits + string.ascii_letters


    def int2base(x, base, pad_to_len):
        if x < 0:
            sign = -1
        elif x == 0:
            return '0'*pad_to_len
        else:
            sign = 1
    
        x *= sign
        digits = []
    
        while x:
            digits.append(digs[x % base])
            x = x // base
    
        if sign < 0:
            digits.append('-')
    
        digits.reverse()
        ret = ''.join(digits)
        if len(ret) < pad_to_len:
            ret = '0'*(pad_to_len-len(ret))+ret
    
        return ret
    
    largest = 0
    for i in range(n_actions**(n_states*horizon)):
        pi_y1 = np.array(list((int2base(i,n_actions,horizon*n_states))),dtype=int).reshape((horizon,n_states))
        for j in range(i,n_actions**(n_states*horizon)):
            pi_y2 = np.array(list((int2base(j,n_actions,horizon*n_states))),dtype=int).reshape((horizon,n_states))
            y1 = vectorize_policy(pi_y1, transitions, init_state_dist, horizon, n_states, n_actions)
            y2 = vectorize_policy(pi_y2, transitions, init_state_dist, horizon, n_states, n_actions)
            if (y1-y2).T @ design_inv @ (y1-y2) > largest:
                largest = (y1-y2).T @ design_inv @ (y1-y2) 
    return largest


def vectorize_policy(
        pi: np.ndarray, transitions: np.ndarray,
        init_state_dist: np.ndarray,
        horizon: int,
        n_states: int,
        n_actions: int,
        ):

    y = np.zeros((horizon*n_states*n_actions))


    one_hot_actions = np.eye(n_actions)

    y_prev_state_action_dist = (one_hot_actions[pi[0]] * init_state_dist[:,None]).reshape(-1)
    y[:n_states*n_actions] = y_prev_state_action_dist

    transitions = transitions.reshape((horizon, n_states*n_actions, n_states))

    for h in range(1, horizon):
        #for h in range(1, 2):
        y_current_state_dist = y_prev_state_action_dist.T @ transitions[h-1]

        y_current_state_action_dist = (one_hot_actions[pi[h]] * y_current_state_dist[:, None]).reshape(-1)

        y[h*n_states*n_actions:(h+1)*n_states*n_actions] = y_current_state_action_dist

        y_prev_state_action_dist = y_current_state_action_dist

    return y








class Optimizer:
    """Implements the convex optimization problem at the core of AceIRL."""

    def __init__(
        self,
        delta: float = 0.1,
    ):
        self.delta = delta

    def _solve_opt_problem(
            self,
            y1: cp.Variable,
            y2: cp.Variable,
            y1_det,
            y2_det,
            b: np.ndarray,
            A: np.ndarray,
            v: np.ndarray
            ):

        # Objective
        f = (y1-y2).T@v
        #f = 1
        objective = cp.Maximize(f)

        # Constraints
        constraints = [A @ y1 == b, y1 >= 0, A @ y2 == b, y2 >= 0]
        constraints += [A @ y1_det == b, y1_det >= 0, A @ y2_det == b, y2_det >= 0]
        constraints += [y1_det.T@np.eye(400)@y1_det >= 40]

        #if use_eps_const:
        #    c_ = cp.vstack([c, np.ones((1, 1))])
        #    constraints.append(A1.T @ y_ >= c_)
        #else:
        #    constraints.append(A3.T @ y_ >= c)

        # Define the problem
        prob = cp.Problem(objective, constraints)

        # Solve thr problem 
        eps = prob.solve()
        #obj_value = prob.solve(gp=True)
        # may vary
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
        mu1 = cp.Variable(mu_size, pos=True)
        mu2 = cp.Variable(mu_size, pos=True)
        mu1_det = cp.Variable(mu_size, pos=True)
        mu2_det = cp.Variable(mu_size, pos=True)
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
        return mu1, mu2, mu1_det, mu2_det, bb_, A2


