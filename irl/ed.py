import sys
import copy
import matplotlib.pyplot as plt
sys.path.insert(0, '..')

import numpy as np
import torch
np.set_printoptions(suppress=True)
import argparse
import tqdm
import itertools

from envs import Gridworld, driver_mdp, llm_mdp, llm_simple_mdp  # removed river swim imports
from typing import Tuple, Optional
from helpers import (sample_trajectory_from_visitation, format_func,
                    vectorize_policy, vectorize_uniform_policy, to_torch, set_helpers_seed)
from rl.value_iteration import value_iteration, policy_evaluation
from multiprocessing import Pool
from functools import partial
from sklearn.linear_model import LogisticRegression
from multinomial_objectives import (
    func_orig,  func_orig_grad,
    func_agg, func_agg_grad,
    func_naive, func_naive_grad, 
)
#from multinomial_objectives import func_orig_tr as func_orig
#from multinomial_objectives import func_orig_tr_grad as func_orig_grad


def transform_and_plot_d1(d1, shape):
    horizon, n_states, n_actions = shape
    # Reshape and sum over actions and horizon
    length = shape[1] // shape[2]  # Calculate n_lanes from shape
    n_lanes = shape[2]  # Get length from shape
    d1_reshaped = d1.reshape(horizon, n_lanes, length, n_actions).sum(dim=-1)
    result = d1_reshaped.sum(dim=0)
    
    # Normalize the result
    result /= horizon
    #result 
    
    # Plot the result
    plt.figure(figsize=(12, 6))
    plt.imshow(result.T, cmap='Reds', aspect='auto', origin='lower')  # Transpose the result for plotting
    plt.colorbar(label='Probability')
    plt.title('Driver Environment State Distribution (Summed over Horizon)')
    plt.xlabel('Lanes')
    plt.ylabel('Length')
    plt.xticks(range(n_lanes))
    plt.yticks(range(length))
    plt.show()
    
    return result

def get_X_trajs(d_list, phi, mdp, init_state_dist, sparse, K):
    horizon, n_states, n_actions, _ = mdp.shape
    X_list = []
    traj_list = []
    
    is_state_only = phi.shape[0] == n_states
    
    for d in d_list:
        states, actions = to_torch(sample_trajectory_from_visitation(d, mdp, init_state_dist))
        
        # Calculate indices
        if is_state_only:
            X = phi[states[:-1].int()]
        else:
            # For state-action features, use combined indices directly
            X = phi[(states[:-1] * n_actions + actions).int()]
            
        # Get trajectory one-hot encoding
        eye = torch.eye(n_states*n_actions)
        traj = eye[(states[:-1] * n_actions + actions).int()]
        
        X_list.append(X)
        traj_list.append(traj)
    
    return X_list, traj_list


def calculate_objective(phi, d_list, traj_mix_list, t, adaptive, func, horizon, n_states, n_actions, T, lambda_reg):
    """
    Calculate objective value for K trajectories
    Args:
        phi: Feature matrix
        d_list: List of K visitation distributions or None
        traj_mix_list: List of K trajectory mixtures
        t: Current iteration (-1 for final calculation)
        adaptive: Whether to use adaptive updates
        func: Objective function to compute value
        horizon, n_states, n_actions, T: Environment parameters
    """
    if d_list is None or t == -1:
        return func(traj_mix_list, phi, horizon, n_states, n_actions, T, lambda_reg)

    if adaptive:
        # Create adapted trajectory list using mixture of current and running average
        adapted_trajs = []
        for i, (d, traj_mix) in enumerate(zip(d_list, traj_mix_list)):
            adapted_trajs.append(d/(t+1) + traj_mix*t/(t+1))
        return func(adapted_trajs, phi, horizon, n_states, n_actions, T, lambda_reg)
    
    return func(d_list, phi, horizon, n_states, n_actions, T, lambda_reg)
       
def algorithm(phi, mdp, reward, init_state_dist, T, n_opt_iters, adaptive, obj_func, grad_func, sparse, rand_init, lambda_reg, K=2):
    """
    Algorithm modified to handle K-way comparisons
    K: number of trajectories to compare (default=2 for backward compatibility)
    """
    horizon, n_states, n_actions, _ = mdp.shape

    # Initialize lists to store K different X and trajectory mixtures
    X_list = [None for _ in range(K)]
    traj_mix_list = [torch.zeros((horizon, n_states, n_actions)) for _ in range(K)]

    for t in range(T):
        if adaptive or t==0 or rand_init:
            # Get K different d values from optimize_func
            d_list = optimize_func(phi, traj_mix_list, mdp, reward, init_state_dist, t, n_opt_iters, adaptive, obj_func, grad_func, T, K, lambda_reg=lambda_reg)
            #transform_and_plot_d1(d_list[0], mdp.shape[:3])  # Just plotting first d for now

        # Get K different X and traj values
        X_t_list, traj_list = get_X_trajs(d_list, phi, mdp, init_state_dist, sparse, K)
        
        # Reshape all trajectories
        traj_list = [traj.view((horizon, n_states, n_actions)) for traj in traj_list]

        # Update all trajectory mixtures
        for k in range(K):
            traj_mix_list[k] = traj_mix_list[k]*t/(t+1) + traj_list[k]/(t+1)

        # Store or concatenate X values
        if t == 0:
            X_list = X_t_list
        else:
            X_list = [torch.concatenate([X_list[k], X_t_list[k]]) for k in range(K)]

    # Reshape all X values
    X_list = [X.reshape((T, horizon, -1)).permute(1,0,2) for X in X_list]
    X_per_h_list = [torch.stack([X[i,:,:] for i in range(horizon)]) for X in X_list]
    
    objective_val_trajs_orig = calculate_objective(phi, None, traj_mix_list, -1, False, func_orig, horizon, n_states, n_actions, T, lambda_reg)
    
    return X_per_h_list, objective_val_trajs_orig

def optimize_func(phi, traj_mix_list, mdp, reward, init_state_dist, t, n_opt_iters, adaptive, obj_func, grad_func, T, K, lambda_reg):
    """
    Optimize function for K trajectories
    Args:
        phi: Feature matrix
        traj_mix_list: List of K trajectory mixtures
        mdp: MDP object
        reward: Reward function
        init_state_dist: Initial state distribution
        t: Current iteration
        n_opt_iters: Number of optimization iterations
        adaptive: Whether to use adaptive updates
        grad_func: Gradient function
        T: Number of trajectories
        K: Number of trajectories to compare
    """
    horizon, n_states, n_actions, _ = mdp.shape
    
    # Initialize K random visitation distributions
    d_random_list = [
        vectorize_uniform_policy(mdp, init_state_dist, horizon, n_states, n_actions, random_policy=n_opt_iters > 0)
        for _ in range(K)
    ]
    # Initialize K distributions to optimize
    d_list = [d_random.clone().detach() for d_random in d_random_list]
    
    # If using original gradient function, first optimize with aggregate objective
    if grad_func == func_orig_grad:
        d_list = optimize_func(phi, traj_mix_list, mdp, reward, init_state_dist, t, 
                             n_opt_iters // 2, adaptive, func_agg, func_agg_grad, T, K, lambda_reg)
        
    n_rounds = 4  # Number of rounds to repeat the optimization
    learning_rate = 0.1
    iters_per_round = n_opt_iters // n_rounds  # Split iterations across rounds
    
    for round in range(n_rounds):
        # Optimize each distribution in sequence
        for k in range(K):
            # Optimize current distribution k for iters_per_round iterations
            for i in range(iters_per_round):
                if adaptive:
                    # Create convex combination for gradient evaluation
                    mixed_d_list = []
                    for d, traj_mix in zip(d_list, traj_mix_list):
                        traj_mix_flat = traj_mix.reshape(-1)  # same shape as d
                        mixed_d = (t/(t+1)) * traj_mix_flat + (1/(t+1)) * d
                        mixed_d_list.append(mixed_d)
                    grad_list = grad_func(mixed_d_list, phi, horizon, n_states, n_actions, T, lambda_reg)
                    # Scale gradients by (t+1) to maintain consistent learning rate
                    grad_list = [grad * (t+1) for grad in grad_list]
                else:
                    grad_list = grad_func(d_list, phi, horizon, n_states, n_actions, T, lambda_reg)
                
                # Only use gradient for current distribution k
                grad_k = grad_list[k]
                d_reward = -grad_k.reshape((horizon, n_states, n_actions))
                
                # Compute optimal policy for this reward
                pi_opt = value_iteration(mdp, d_reward, random_argmax_flag=False)[2]
                
                # Compute visitation distribution for this policy
                d_opt = vectorize_policy(pi_opt, mdp, init_state_dist, horizon, n_states, n_actions)
                
                # Update only the k-th distribution using gradient descent
                d_list[k] = (1 - learning_rate) * d_list[k] + learning_rate * d_opt.clone().detach()
    
    return d_list



def setup_env(args):
    """Set up environment and return components needed for algorithm"""
    torch.manual_seed(args.seed)

    if args.env == 'gridworld':
        size = 3
        grid = Gridworld(size=size, horizon=6, p_fail=0.0, p_obst=1.0, seed=args.seed)
        mdp, r_orig, init_state_dist = grid.get_mdp()
        
        d = 4
        phi = torch.normal(torch.zeros((size**2, d)), 0.5*torch.ones((size**2, d)))
        theta = torch.ones(d) - 0.5
        r = phi @ theta
        r = r.reshape(r_orig.shape)

    elif args.env == 'driver':
        mdp, r_orig, init_state_dist, phi = driver_mdp(n_lanes=5, length=12, horizon=12)
        d = 2  # dimension of phi is 2
        theta = torch.ones(d)  # theta is all ones vector
        theta[0] = 0
        r = torch.tensor(phi).float() @ theta  # calculate r using phi and theta
        r = r.reshape(r_orig.shape)

    elif args.env == 'llm':
        result = llm_simple_mdp(
            token_file=args.token_file,
            horizon=args.horizon,
            seed=args.seed,
            device=args.device if hasattr(args, 'device') else 'cpu',
            pca_dim=args.pca_dim
        )
        
        if result is None:
            return None
            
        mdp, theta, init_state_dist, phi = result
        phi = phi.float()
        # Normalize each row of phi to have L2 norm of 1
        phi = phi / torch.norm(phi, p=2, dim=1, keepdim=True)
        theta = theta.float()
        theta = theta / torch.norm(theta, p=1)

    return to_torch([mdp, theta, init_state_dist, phi])

def run_algorithm(env_components, args):
    """Run algorithm with provided environment components"""
    mdp, theta, init_state_dist, phi = env_components
    horizon, n_states, n_actions, _ = mdp.shape

    obj_func, grad_func = select_funcs(args.grad_func)

    torch.manual_seed(args.seed)  # Reset seed for algorithm randomness
    set_helpers_seed(args.seed)
    
    X_per_h_list, objective_val_trajs_orig = algorithm(
        phi, mdp, theta, init_state_dist, args.T, args.n_opt_iters, 
        args.adaptive, obj_func, grad_func, args.sparse, 
        args.rand_init, args.lambda_reg, K=args.K
    )

    # Rest of the current main() function...
    # Initialize lists to store preferences for each horizon
    y_per_h = []

    # Process each horizon separately
    for h in range(horizon):
        X_h = torch.stack([X_per_h_list[k][h] for k in range(args.K)])
        logits_h = X_h @ theta  
        probs_h = torch.nn.functional.softmax(logits_h, dim=0)
        
        y_h = torch.zeros_like(probs_h)
        for t in range(args.T):
            choice = torch.multinomial(probs_h[:, t], 1)
            y_h[choice, t] = 1
        
        y_per_h.append(y_h)

    if len(torch.unique(torch.cat([y.flatten() for y in y_per_h]))) == 1:
        theta_hat = torch.zeros_like(theta)
    else:
        X = torch.cat([X_per_h_list[k].reshape(horizon * args.T, -1) for k in range(args.K)])
        
        y = []
        for k in range(args.K):
            y_k = []
            for h in range(horizon):
                y_k.extend(y_per_h[h][k].tolist())
            y.extend(y_k)
        y = torch.tensor(y)
        
        model = LogisticRegression(
            penalty='l2',
            C=1.0 / args.lambda_reg,
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=False
        )
        model.fit(X, y)
        
        theta_hat = model.coef_.flatten()

    # Compute true reward
    r = phi @ theta
    r = r.reshape(n_states, n_actions)

    # Compute estimated reward
    r_hat = phi @ theta_hat
    r_hat = r_hat.reshape(r.shape)

    # Compute alignment error
    r_hat_flat = r_hat.reshape(-1)
    r_flat = r.reshape(-1)
    
    idx = torch.arange(len(r_flat))
    K_way_combinations = torch.combinations(idx, r=args.K)
    alignment_err = 0
    
    for combo in K_way_combinations:
        true_rewards = r_flat[combo]
        est_rewards = r_hat_flat[combo]
        
        true_probs = torch.nn.functional.softmax(true_rewards, dim=0)
        est_probs = torch.nn.functional.softmax(est_rewards, dim=0)
        
        alignment_err += torch.sum((true_probs - est_probs) ** 2)
    
    alignment_err = alignment_err.float() / (2*len(K_way_combinations))
    #import ipdb; ipdb.set_trace()

    return alignment_err, objective_val_trajs_orig

def is_deterministic_env(env_name):
    """Check if environment is deterministic"""
    return env_name in ['driver', 'llm']

def collect_data(T_values, args):
    means_mse = []
    std_devs_mse = []
    means_orig_val = []
    std_devs_orig_val = []
    
    # For deterministic environments, load once with initial seed
    env_components = None
    if is_deterministic_env(args.env):
        args.seed = 1  # Set initial seed for environment setup
        env_components = setup_env(args)
        if env_components is None:  # Handle llm cache case
            return None, None, None, None

    for T in T_values:
        args.T = T  # Set T for this round of trials
        if args.debug:
            if env_components:  # Deterministic env
                results = []
                for seed in tqdm.tqdm(range(1, args.num_samples + 1)):
                    args.seed = seed  # Update seed for each trial
                    results.append(run_algorithm(env_components, args))
            else:  # Non-deterministic env
                results = [run_single_trial(args, T, seed) 
                          for seed in tqdm.tqdm(range(1, args.num_samples + 1))]
        else:
            with Pool(4) as pool:
                if env_components:  # Deterministic env
                    # Create args copies with different seeds
                    trial_args = []
                    for seed in range(1, args.num_samples + 1):
                        seed_args = copy.deepcopy(args)
                        seed_args.seed = seed
                        trial_args.append(seed_args)
                    run_trial = partial(run_algorithm, env_components)
                    results = list(tqdm.tqdm(pool.imap(run_trial, trial_args), 
                                           total=args.num_samples))
                else:  # Non-deterministic env
                    run_trial = partial(run_single_trial, args, T)
                    results = list(tqdm.tqdm(pool.imap(run_trial, 
                                 range(1, args.num_samples + 1)), 
                                 total=args.num_samples))

        if results[0] is None:
            return None, None, None, None
            
        mses, orig_obj_vals = zip(*results)
        
        means_mse.append(np.mean(mses))
        std_devs_mse.append(np.std(mses))
        means_orig_val.append(np.mean(orig_obj_vals))
        std_devs_orig_val.append(np.std(orig_obj_vals))
    
    return means_mse, std_devs_mse, means_orig_val, std_devs_orig_val


def run_single_trial(args, T, seed):
    """For non-deterministic environments"""
    args.T = T  # No need for _replace since we're mutating args directly
    args.seed = seed
    env_components = setup_env(args)
    if env_components is None:
        return None
    return run_algorithm(env_components, args)
    
def select_funcs(func_name):
    if func_name is None:
        return func_orig, func_orig_grad  # Use any funcs for random method since they won't be used
    elif func_name == 'agg':
        return func_agg, func_agg_grad
    elif func_name == 'naive':
        return func_naive, func_naive_grad
    elif func_name == 'orig':
        return func_orig, func_orig_grad
    else:
        raise ValueError(f"Unknown function type: {func_name}")


def plot_results(T_values, num_samples, data_sets, y_label):
    plt.figure(figsize=(12, 7))
    
    method_colors = {
        'random': 'blue',
        'agg': 'orange',
        'naive': 'red',
        'orig': 'green'
    }
    method_styles = {
        'random': '-',
        'agg': '--',
        'naive': '-.',
        'orig': ':'
    }
    colors = [method_colors[method] for method in methods_to_run if method in results]
    line_styles = [method_styles[method] for method in methods_to_run if method in results]
    
    plt.rcParams.update({'font.size': 16})  # Increase overall font size
    all_values = []
    
    for (means, std_devs), color, label, ls in zip(data_sets, colors, labels, line_styles):
        plt.plot(T_values, means, label=label, color=color, linestyle=ls, linewidth=4) 
        plt.fill_between(T_values, 
                     np.array(means) - 1 * np.array(std_devs) / int(np.sqrt(num_samples)), 
                     np.array(means) + 1 * np.array(std_devs) / int(np.sqrt(num_samples)), 
                     color=color, alpha=0.2)
    
        all_values.extend(means)
    plt.xlabel('T', fontsize=18)
    plt.ylabel(y_label, fontsize=24)
    plt.xscale('log')
    plt.xticks(T_values, [str(T) for T in T_values], fontsize=22)

    if y_label == r'$\text{PME}(r, \hat{r})$':
        plt.yscale('log')
        min_val, max_val = min(all_values), max(all_values)
        
        # Fixed ticks that work well for normalized data (0-1 range)
        tick_locations = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        plt.ylim(0.001, 1.0)  # Start from a very small positive number for log scale
        plt.yticks(tick_locations, [f"{tick:.3f}" for tick in tick_locations], fontsize=24)
    else:
        plt.yticks(fontsize=24)
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=26)
    
    plt.tight_layout()  # Adjust the layout to prevent cut-off labels
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Experimental Design for PBRL')
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--adaptive', action='store_true', dest='adaptive', default=False)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--rand_init', action='store_true')
    parser.add_argument('--grad_func', type=str, default='agg', 
                    help='Which gradient function to use (agg, naive, orig)')
    parser.add_argument('--n_opt_iters', type=int, default=30)
    parser.add_argument('-T', nargs='+', type=int)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--n_actions', type=int)
    parser.add_argument('--n_states', type=int)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--goal_threshold', type=int)
    parser.add_argument('-env', required=True)
    parser.add_argument('--methods', nargs='+', choices=['random', 'agg', 'naive', 'orig'],
                        help='Methods to run (default: random, naive, orig)')

    parser.add_argument('--K', type=int, default=2,
                    help='Number of trajectories to compare (default: 2)')
    parser.add_argument('--token_file', type=str, help='File containing tokens/words')
    parser.add_argument('--device', type=str, default='cpu', help='Device for CLIP model')
    parser.add_argument('--lambda_reg', type=float, default=1.0, help='Regularization parameter for gradient functions')
    parser.add_argument('--pca_dim', type=int, default=50, help='Number of PCA components (use 768 for original dimension)')
    
    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()
    T_values = args.T if args.T else [1, 2, 4, 8, 16]
    #T_values = [512]

    # Define method configurations
    method_configs = {
        'random': {'n_opt_iters': 0, 'grad_func': None, 'label': 'Random Exploration'},
        'agg': {'n_opt_iters': 100, 'grad_func': 'agg', 'label': 'ED-PBRL (agg)'},
        'naive': {'n_opt_iters': 100, 'grad_func': 'naive', 'label': 'ED-PBRL (naive)'},
        'orig': {'n_opt_iters': 150, 'grad_func': 'orig', 'label': 'ED-PBRL (orig)'}
    }

    # Get methods to run from arguments or use default
    methods_to_run = args.methods if hasattr(args, 'methods') else ['random', 'naive', 'orig']
    
    # Collect results for each method
    results = {}
    for method in methods_to_run:
        if method not in method_configs:
            print(f"Warning: Unknown method {method}, skipping...")
            continue
            
        config = method_configs[method]
        args.n_opt_iters = config['n_opt_iters']
        args.grad_func = config['grad_func']
        
        means_mse, std_devs_mse, means_orig_val, std_devs_orig_val = collect_data(T_values, args)
        if means_mse == None:
            sys.exit(0)

        results[method] = {
            'mse': (means_mse, std_devs_mse),
            'obj_val': (means_orig_val, std_devs_orig_val),
            'label': config['label']
        }
    
    # Prepare data for plotting
    mse_data = [(results[method]['mse'][0], results[method]['mse'][1]) 
                for method in methods_to_run if method in results]
    obj_val_data = [(results[method]['obj_val'][0], results[method]['obj_val'][1]) 
                    for method in methods_to_run if method in results]
    
    # Update plot_results function call to use method labels
    labels = [results[method]['label'] for method in methods_to_run if method in results]
    
    # Plot results
    plot_results(T_values, args.num_samples, mse_data, r'$\text{PME}(r, \hat{r})$')
    plot_results(T_values, args.num_samples, obj_val_data, r'$U(d^1,d^2)$')
