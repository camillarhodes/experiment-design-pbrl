import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '..')

import numpy as np
import torch
np.set_printoptions(suppress=True)
import argparse
import tqdm
import itertools

from envs import random_mdp, Gridworld, chain_mdp, river_swim_mdp, two_river_swim_mdp, modified_river_swim_mdp

from typing import Tuple, Optional
from helpers import (sample_trajectory_from_visitation, format_func,
                     vectorize_policy, vectorize_random_policy, to_torch, entropy)
from rl.value_iteration import value_iteration, policy_evaluation
from multiprocessing import Pool
from functools import partial



LAMBDA = 1


def algorithm(phi, mdp, reward, init_state_dist, T, n_opt_iters, adaptive, grad_func, sparse, rand_init):


    horizon, n_states, n_actions, _ = mdp.shape

    X1 = None
    X2 = None

    traj1_mix = torch.zeros((horizon, n_states, n_actions))
    traj2_mix = torch.zeros((horizon, n_states, n_actions))

    for t in range(T):
        if adaptive or t==0 or rand_init:
            d1, d2 = optimize_func(phi, traj1_mix, traj2_mix, mdp, reward, init_state_dist, t, n_opt_iters, adaptive, grad_func, T)
            import ipdb; ipdb.set_trace()

        #import ipdb; ipdb.set_trace()
        X1_t, X2_t, traj1, traj2 = get_X1_X2_trajs(d1, d2, phi, mdp, init_state_dist, sparse)
        traj1 = traj1.view((horizon, n_states, n_actions))
        traj2 = traj2.view((horizon, n_states, n_actions))

        traj1_mix = traj1_mix*t/(t+1) + traj1/(t+1)
        traj2_mix = traj2_mix*t/(t+1) + traj2/(t+1)

        if t == 0:
            X1 = X1_t
            X2 = X2_t
        else:
            X1 = torch.concatenate([X1, X1_t])
            X2 = torch.concatenate([X2, X2_t])

    X1 = X1.reshape((T, horizon, -1)).permute(1,0,2)
    X2 = X2.reshape((T, horizon, -1)).permute(1,0,2)
    X1_per_h = torch.stack([X1[i,:,:] for i in range(horizon)])
    X2_per_h = torch.stack([X2[i,:,:] for i in range(horizon)])
    X_per_h = (X1_per_h - X2_per_h).reshape(horizon, T, -1)
    #X_per_h = (X1_per_h[:, :, np.newaxis, :] - X2_per_h[:, np.newaxis, :, :]).reshape(horizon, T**2, -1)

    #if not sparse:
    #    y = y.reshape((T, horizon)).permute(1,0)
    #    y_per_h = [y[i,:] for i in range(horizon)]
    #    y = torch.cat(y_per_h)

    objective_val_trajs = calculate_objective(phi, None, None, traj1_mix, traj2_mix, -1, False, func_approx, horizon, n_states, n_actions, T)
    #if T >= 64:
    #    import ipdb; ipdb.set_trace()

    objective_val_trajs_orig = calculate_objective(phi, None, None, traj1_mix, traj2_mix, -1, False, func_orig, horizon, n_states, n_actions, T)

    
    return X_per_h, objective_val_trajs_orig, objective_val_trajs

def get_X1_X2_trajs(d1, d2, phi, mdp, init_state_dist,  sparse):
    horizon, n_states, n_actions, _ = mdp.shape
    states1, actions1 = to_torch(sample_trajectory_from_visitation(d1, mdp, init_state_dist)) 
    states2, actions2 = to_torch(sample_trajectory_from_visitation(d2, mdp, init_state_dist) )

    # action indices
    act_indices1 = (states1[:-1] * n_actions + actions1).int()
    act_indices2 = (states2[:-1] * n_actions + actions2).int()

    state_indices1 = states1[:-1].int()
    state_indices2 = states2[:-1].int()



    X1 = phi[state_indices1]
    X2 = phi[state_indices2]

    eye = torch.eye(n_states*n_actions)
    traj1 = eye[act_indices1]
    traj2 = eye[act_indices2]

    return X1, X2, traj1, traj2


def get_X_and_y_and_trajs(d1, d2, phi, mdp, init_state_dist, prefs_mat, sparse):
    
    
    X = phi[indices1] - phi[indices2]
    prefs = prefs_mat[indices1, indices2]
    if sparse:
        pref = prefs.sum() / horizon
        y = 2*torch.bernoulli(pref) - 1
    else:
        y = 2*torch.bernoulli(prefs) - 1

    return X, y, traj1, traj2


def func_approx(d1, d2, phi, horizon, n_states, n_actions, T):
    d1 = d1.view(horizon, n_states, n_actions).sum(dim=2)
    d2 = d2.view(horizon, n_states, n_actions).sum(dim=2)
    sigma_inv = torch.zeros((len(phi.T), len(phi.T)))

    for h in range(horizon):
        sigma_inv += T * phi.T @ torch.diag(d1[h] + d2[h] - 2 * d1[h] * d2[h]) @ phi

    sigma_inv += LAMBDA * torch.eye(len(phi.T))

    return -torch.log(sigma_inv.det())

def func_approx_grad(d1, d2, phi, horizon, n_states, n_actions, T):
    d1 = d1.view(horizon, n_states, n_actions)
    d2 = d2.view(horizon, n_states, n_actions)
    
    d1_sum = d1.sum(dim=2)
    d2_sum = d2.sum(dim=2)
    
    v = d1_sum + d2_sum - 2*d1_sum * d2_sum
    
    sigma_inv = T * sum(phi.T @ torch.diag(v[h]) @ phi for h in range(horizon))
    sigma_inv += LAMBDA * torch.eye(phi.shape[1], device=d1.device)
    
    M = torch.inverse(sigma_inv)
    
    phi_M_phi_T = phi @ M @ phi.T
    
    grad_d1 = -T * (1 - 2*d2_sum).unsqueeze(2).expand_as(d1) * phi_M_phi_T.diag().unsqueeze(0).unsqueeze(2)
    grad_d2 = -T * (1 - 2*d1_sum).unsqueeze(2).expand_as(d2) * phi_M_phi_T.diag().unsqueeze(0).unsqueeze(2)
    
    grad_d1 = grad_d1.reshape(-1)
    grad_d2 = grad_d2.reshape(-1)
    
    return grad_d1, grad_d2

def func_no_interactions(d1, d2, phi, horizon, n_states, n_actions, T):
    d1 = d1.view(horizon, n_states, n_actions).sum(dim=2)
    d2 = d2.view(horizon, n_states, n_actions).sum(dim=2)
    sigma_inv = torch.zeros((len(phi.T), len(phi.T)))

    for h in range(horizon):
        sigma_inv += T * phi.T @ torch.diag(d1[h] + d2[h]) @ phi

    sigma_inv += LAMBDA * torch.eye(len(phi.T))

    return -torch.log(sigma_inv.det())

def func_no_interactions_grad(d1, d2, phi, horizon, n_states, n_actions, T):
    d1 = d1.view(horizon, n_states, n_actions)
    d2 = d2.view(horizon, n_states, n_actions)
    
    d1_sum = d1.sum(dim=2)
    d2_sum = d2.sum(dim=2)
    
    v = d1_sum + d2_sum
    
    sigma_inv = T * sum(phi.T @ torch.diag(v[h]) @ phi for h in range(horizon))
    sigma_inv += LAMBDA * torch.eye(phi.shape[1], device=d1.device)
    
    M = torch.inverse(sigma_inv)
    
    phi_M_phi_T = phi @ M @ phi.T
    
    grad_d1 = -T * phi_M_phi_T.diag().unsqueeze(0).unsqueeze(2).expand_as(d1)
    grad_d2 = -T * phi_M_phi_T.diag().unsqueeze(0).unsqueeze(2).expand_as(d2)
    
    grad_d1 = grad_d1.reshape(-1)
    grad_d2 = grad_d2.reshape(-1)
    
    return grad_d1, grad_d2

def func_orig(d1, d2, phi, horizon, n_states, n_actions, T):
    d1 = d1.view(horizon, n_states, n_actions).sum(dim=2)
    d2 = d2.view(horizon, n_states, n_actions).sum(dim=2)
    sigma_inv = torch.zeros((len(phi.T), len(phi.T)))

    for h in range(horizon):
        sigma_inv += T * phi.T @ (torch.diag(d1[h] + d2[h]) - d1[h][:, None] @ d2[h][:, None].T - d2[h][:, None] @ d1[h][:, None].T) @ phi

    sigma_inv += LAMBDA * torch.eye(len(phi.T))

    return -torch.log(sigma_inv.det())

def func_orig_grad(d1, d2, phi, horizon, n_states, n_actions, T):
    d1 = d1.view(horizon, n_states, n_actions)
    d2 = d2.view(horizon, n_states, n_actions)
    
    d1_sum = d1.sum(dim=2)
    d2_sum = d2.sum(dim=2)
    
    sigma_inv = torch.zeros((phi.shape[1], phi.shape[1]), device=d1.device)
    for h in range(horizon):
        sigma_inv += T * phi.T @ (torch.diag(d1_sum[h] + d2_sum[h]) - 
                                  d1_sum[h][:, None] @ d2_sum[h][:, None].T - 
                                  d2_sum[h][:, None] @ d1_sum[h][:, None].T) @ phi
    sigma_inv += LAMBDA * torch.eye(phi.shape[1], device=d1.device)
    
    M = torch.inverse(sigma_inv)
    
    phi_M_phi_T = phi @ M @ phi.T
    
    grad_d1 = torch.zeros_like(d1)
    grad_d2 = torch.zeros_like(d2)
    
    for h in range(horizon):
        grad_d1[h] -= T * phi_M_phi_T.diag().unsqueeze(1)
        grad_d2[h] -= T * phi_M_phi_T.diag().unsqueeze(1)
        
        grad_d1[h] += T * (phi_M_phi_T @ d2_sum[h] + phi_M_phi_T.T @ d2_sum[h]).unsqueeze(1)
        grad_d2[h] += T * (phi_M_phi_T @ d1_sum[h] + phi_M_phi_T.T @ d1_sum[h]).unsqueeze(1)
    
    grad_d1 = grad_d1.reshape(-1)
    grad_d2 = grad_d2.reshape(-1)
    
    return grad_d1, grad_d2

def calculate_objective(phi, d1, d2, traj1_mix, traj2_mix, t, adaptive, func, horizon, n_states, n_actions, T):

    if d1 == None or d2 == None or t == -1:
        return func(traj1_mix, traj2_mix, phi, horizon, n_states, n_actions, T)

    if adaptive:
        return func(d1/(t+1)+traj1_mix*t/(t+1), d2/(t+1)+traj2_mix*t/(t+1), phi, horizon, n_states, n_actions, T)
    return func(d1, d2, phi, horizon, n_states, n_actions, T)
        
    

def optimize_func(phi, traj1_mix, traj2_mix, mdp, reward, init_state_dist, t, n_opt_iters, adaptive, grad_func, T):
    horizon, n_states, n_actions, _ = mdp.shape
    learning_rate = 0.05
    
    d_random = vectorize_random_policy(mdp, init_state_dist, horizon, n_states, n_actions)
    d1 = d_random.clone().detach()
    d2 = d_random.clone().detach()    
    # Choose the appropriate gradient function
    
    for i in range(n_opt_iters):
        # Compute the gradients analytically
        grad_d1, grad_d2 = grad_func(d1, d2, phi, horizon, n_states, n_actions, T)
        
        # Reshape gradients
        d1_reward = -grad_d1.reshape((horizon, n_states, n_actions))
        d2_reward = -grad_d2.reshape((horizon, n_states, n_actions))
        
        assert d1_reward.shape == (horizon, n_states, n_actions)
        
        pi1_opt = value_iteration(mdp, d1_reward, random_argmax_flag=False)[2]
        pi2_opt = value_iteration(mdp, d2_reward, random_argmax_flag=False)[2]
        
        d1_opt = vectorize_policy(pi1_opt, mdp, init_state_dist, horizon, n_states, n_actions)
        d2_opt = vectorize_policy(pi2_opt, mdp, init_state_dist, horizon, n_states, n_actions)
        #import ipdb; ipdb.set_trace()
        
        # Update the variables using gradient descent
        #if i % 2 == 1:
        #    d1 = (1 - learning_rate) * d1 + learning_rate * d1_opt.clone().detach()
        #else:
        #    d2 = (1 - learning_rate) * d2 + learning_rate * d2_opt.clone().detach()

        d1 = (1 - learning_rate) * d1 + learning_rate * d1_opt.clone().detach()
        d2 = (1 - learning_rate) * d2 + learning_rate * d2_opt.clone().detach()
        
        # Print the function value and the variables at each iteration
        if i % 2 == 0:
            # Calculate the objective only when we're about to print it
            with torch.no_grad():
                z = calculate_objective(phi, d1, d2, traj1_mix, traj2_mix, t, False, func_orig, horizon, n_states, n_actions, T)
                print(f"Iteration {i+1}: z = {z.item():.4f}")
    
    return (d1, d2)

def main(args):
    if args.env == 'random':
        torch.manual_seed(args.seed)
        mdp, r_orig, init_state_dist = random_mdp(args.horizon, args.n_states, args.n_actions, seed=args.seed)
        horizon, n_states, n_actions, _ = mdp.shape
        d = 8
        phi = torch.normal(torch.zeros((n_states, d)),1*torch.ones((n_states,d)))
    # Compute L2 norm for each row
        row_norms = torch.norm(phi, p=2, dim=1, keepdim=True)
        phi = torch.where(row_norms > 1, phi / row_norms, phi)


        theta = torch.rand(d) - 0.5
        theta /= 2*theta.norm()
        r = phi @ theta
        r = r.reshape(r_orig.shape)


    elif args.env == 'gridworld':
        #size = 3
        size = 4
        #phi = torch.eye(size**2)
        grid = Gridworld(size=size, horizon=8, p_fail=0.0, p_obst=1.0, seed=args.seed)
        mdp, r_orig, init_state_dist = grid.get_mdp()
        #grid = Gridworld(size=size, horizon=5, p_fail=0, p_obst=1, seed=args.seed)
        #grid = Gridworld(size=3, horizon=5, p_obst=0, p_fail=0)
        #grid = Gridworld(size=3, horizon=5, d = 2
        torch.manual_seed(args.seed)
        d=4
        phi = torch.normal(torch.zeros((size**2, d)),1*torch.ones((size**2,d)))
    # Compute L2 norm for each row
        row_norms = torch.norm(phi, p=2, dim=1, keepdim=True)
        phi = torch.where(row_norms > 1, phi / row_norms, phi)


        theta = torch.rand(d) - 0.5
        theta /= 2*theta.norm()
        r = phi @ theta
        r = r.reshape(r_orig.shape)


    elif args.env == 'river_swim':
        mdp, r_orig, init_state_dist = river_swim_mdp(args.n_states, args.horizon)
        d=32
        phi = torch.normal(torch.zeros((args.n_states, d)),1*torch.ones((args.n_states,d)))
        row_norms = torch.norm(phi, p=2, dim=1, keepdim=True)
        phi = torch.where(row_norms > 1, phi / row_norms, phi)
        theta = torch.rand(d) - 0.5
        theta /= 2*theta.norm()
        r = phi @ theta
        r = r.reshape(r_orig.shape)

    elif args.env == 'two_river_swim':
        mdp, r_orig, init_state_dist = two_river_swim_mdp(args.n_states, args.horizon)
        horizon, n_states, n_actions, _ = mdp.shape
        d=8
        phi = torch.normal(torch.zeros((n_states, d)),1*torch.ones((n_states,d)))
        row_norms = torch.norm(phi, p=2, dim=1, keepdim=True)
        phi = torch.where(row_norms > 1, phi / row_norms, phi)
        theta = torch.rand(d) - 0.5
        theta /= 2*theta.norm()
        r = phi @ theta
        r = r.reshape(r_orig.shape)

    elif args.env == 'modified_river_swim':
        mdp, r_orig, init_state_dist, phi= modified_river_swim_mdp(args.n_actions, args.n_states, args.horizon)
        horizon, n_states, n_actions, _ = mdp.shape
    
    # Set phi to the identity matrix
        #phi = torch.eye(d)
        
        # Use r_orig as theta
        theta = torch.rand(len(phi.T)) - 0.5
        theta /= 2*theta.norm()
        phi = torch.from_numpy(phi).float()
        
        # Compute r (which should be identical to r_orig in this case)
        r = phi @ theta
        


#    elif args.env == 'alternating':
#        n_actions = 2
#        phi = torch.eye(args.n_states)
#        mdp, r, init_state_dist = alternating_path_mdp(args.n_states, args.horizon)
#

    mdp, r, init_state_dist = to_torch([mdp, r, init_state_dist])

    horizon, n_states, n_actions, _ = mdp.shape

    grad_func = select_grad_func(args.grad_func)


    X_per_h, objective_val_trajs_orig, objective_val_trajs = algorithm(phi, mdp, r, init_state_dist, args.T, args.n_opt_iters, args.adaptive, grad_func, args.sparse, args.rand_init)
    if args.sparse:
        X_stacked = torch.stack(X_per_h, dim=0)
        X = torch.sum(X_stacked, dim=0)
    else:
        #X = X_per_h.reshape(horizon * args.T**2, -1)
        X = X_per_h.reshape(horizon * args.T, -1)
        #prefs = (X @ r.reshape(-1) + 1)/2
        prefs = (X @ theta + 1)/2
        y = 2*torch.bernoulli(prefs) - 1


        theta_hat = torch.linalg.pinv(X.T@X + LAMBDA*torch.eye(len(X.T)))@X.T@y
        r_hat = phi @ theta_hat

    r_hat = r_hat.reshape(r.shape)

    r_hat_flat = r_hat.reshape(-1)
    r_flat = r.reshape(-1)

    # 
    idx = torch.arange(len(r_flat))
    pairs_idx=torch.cartesian_prod(idx,idx)
    alignment_err =torch.pow((r_hat_flat[pairs_idx][:,0] - r_hat_flat[pairs_idx][:,1]) - \
        (r_flat[pairs_idx][:,0] - r_flat[pairs_idx][:,1]),2).sum()
    #if args.T >= 16:
    #    import ipdb; ipdb.set_trace()

    #r = r.unsqueeze(0).repeat(horizon,1,1)
    #r_hat = r_hat.unsqueeze(0).repeat(horizon,1,1)
    #v_opt = value_iteration(mdp, r)[1]
    #pi_hat = value_iteration(mdp, r_hat)[2]
    #v_hat = torch.tensor(value_iteration(mdp, r, pi_hat)[1])
    #alignment_err = v_hat[0]@(init_state_dist).double()
    return alignment_err, objective_val_trajs_orig, objective_val_trajs

def select_grad_func(grad_func_name):
    if grad_func_name == 'approx':
        return func_approx_grad
    elif grad_func_name == 'no_interactions':
        return func_no_interactions_grad
    elif grad_func_name == 'orig':
        return func_orig_grad
    else:
        raise ValueError(f"Unknown gradient function: {grad_func_name}")

def collect_data(T_values, args):
    means_mse = []
    std_devs_mse = []
    means_orig_val = []
    std_devs_orig_val = []
    means_val = []
    std_devs_val = []
    
    for T in T_values:
        if args.debug:
            # Single-process execution
            results = [run_single_trial(args, T, seed) for seed in tqdm.tqdm(range(1, args.num_samples + 1))]
        else:
            # Multiprocessing execution
            with Pool() as pool:
                run_trial = partial(run_single_trial, args, T)
                results = list(tqdm.tqdm(pool.imap(run_trial, range(1, args.num_samples + 1)), total=args.num_samples))


        
        mses, orig_obj_vals, obj_vals = zip(*results)
        
        means_mse.append(np.mean(mses))
        std_devs_mse.append(np.std(mses))
        means_orig_val.append(np.mean(orig_obj_vals))
        std_devs_orig_val.append(np.std(orig_obj_vals))
        means_val.append(np.mean(obj_vals))
        std_devs_val.append(np.std(obj_vals))
    
    return means_mse, std_devs_mse, means_orig_val, std_devs_orig_val, means_val, std_devs_val
# Function to collect data

def plot_results(T_values, num_samples, data_sets, y_label):
    plt.figure(figsize=(12, 7))
    
    colors = ['blue', 'orange', 'red', 'green']
    labels = ['Random Exploration', 'ED-PBRL (Approx)', 'ED-PBRL (No Interactions)', 'ED-PBRL (Orig)']
    line_styles = ['-', '--', '-.', ':']  # Solid, dashed, dash-dot, dotted
    
    plt.rcParams.update({'font.size': 16})  # Increase overall font size
    all_values=[]
    
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
    #plt.yticks(fontsize=24)

    if y_label == r'$\text{PME}(r, \hat{r})$':
        plt.yscale('log')
        #min_val, max_val = min(v for v in all_values if v > 0), max(all_values)
        min_val, max_val = min(all_values), max(all_values)
        
        # Adjust y-axis limits
        plt.ylim(min_val * 0.9, max(max_val * 1.1, 64))
        
        # Fixed ticks as specified
        tick_locations = [0.5, 1, 2, 4, 8, 16, 32, 64]
        plt.yticks(tick_locations, [str(tick) for tick in tick_locations], fontsize=24)
    else:
        plt.yticks(fontsize=24)
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=26)
    
    plt.tight_layout()  # Adjust the layout to prevent cut-off labels
    plt.show()

def run_single_trial(args, T, seed):
    args.T = T
    args.seed = seed
    return main(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Experimental Design for PBRL')
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-adaptive', action='store_false', dest='adaptive')
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--rand_init', action='store_true')
    parser.set_defaults(adaptive=True)
    parser.add_argument('--grad_func', type=str, default='approx', 
                    help='Which gradient function to use (approx, no_interactions, orig)')
    parser.add_argument('--n_opt_iters', type=int, default=30)
    parser.add_argument('-T', type=int, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--n_actions', type=int)
    parser.add_argument('--n_states', type=int)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--goal_threshold', type=int)
    parser.add_argument('-env', required=True)
    args = parser.parse_args()


    T_values = [1, 2, 4, 8, 16, 32, 64]
    T_values = [1]

    # Random Exploration
    args.n_opt_iters = 0
    means_mse_random, std_devs_mse_random, means_orig_obj_val_random, std_devs_orig_obj_val_random, means_obj_val_random, std_devs_obj_val_random = collect_data(T_values, args)
    
    # ED-PBRL (Approx)
    args.n_opt_iters = 150
    args.grad_func = 'approx'
    means_mse_ed_pbrl_approx, std_devs_mse_ed_pbrl_approx, means_orig_obj_val_ed_pbrl_approx, std_devs_orig_obj_val_ed_pbrl_approx, means_obj_val_ed_pbrl_approx, std_devs_obj_val_ed_pbrl_approx = collect_data(T_values, args)
    
    # ED-PBRL (No Interactions)
    args.grad_func = 'no_interactions'
    means_mse_ed_pbrl_no_int, std_devs_mse_ed_pbrl_no_int, means_orig_obj_val_ed_pbrl_no_int, std_devs_orig_obj_val_ed_pbrl_no_int, means_obj_val_ed_pbrl_no_int, std_devs_obj_val_ed_pbrl_no_int = collect_data(T_values, args)
    
    # ED-PBRL (Orig)
    args.grad_func = 'orig'
    means_mse_ed_pbrl_orig, std_devs_mse_ed_pbrl_orig, means_orig_obj_val_ed_pbrl_orig, std_devs_orig_obj_val_ed_pbrl_orig, means_obj_val_ed_pbrl_orig, std_devs_obj_val_ed_pbrl_orig = collect_data(T_values, args)
    
    # Plot MSE results
    mse_data = [
        (means_mse_random, std_devs_mse_random),
        (means_mse_ed_pbrl_approx, std_devs_mse_ed_pbrl_approx),
        (means_mse_ed_pbrl_no_int, std_devs_mse_ed_pbrl_no_int),
        (means_mse_ed_pbrl_orig, std_devs_mse_ed_pbrl_orig)
    ]
    plot_results(T_values, args.num_samples, mse_data, r'$\text{PME}(r, \hat{r})$')
    
    # Plot original objective value results
    orig_obj_val_data = [
        (means_orig_obj_val_random, std_devs_orig_obj_val_random),
        (means_orig_obj_val_ed_pbrl_approx, std_devs_orig_obj_val_ed_pbrl_approx),
        (means_orig_obj_val_ed_pbrl_no_int, std_devs_orig_obj_val_ed_pbrl_no_int),
        (means_orig_obj_val_ed_pbrl_orig, std_devs_orig_obj_val_ed_pbrl_orig)
    ]
    plot_results(T_values, args.num_samples, orig_obj_val_data, r'$U(d^1,d^2)$')
    
    # Plot final objective value results
    obj_val_data = [
        (means_obj_val_random, std_devs_obj_val_random),
        (means_obj_val_ed_pbrl_approx, std_devs_obj_val_ed_pbrl_approx),
        (means_obj_val_ed_pbrl_no_int, std_devs_obj_val_ed_pbrl_no_int),
        (means_obj_val_ed_pbrl_orig, std_devs_obj_val_ed_pbrl_orig)
    ]
    plot_results(T_values, args.num_samples, obj_val_data,  r'$\widetilde{U}(d^1,d^2)$')
