import torch
import numpy as np
from typing import List, Tuple

def func_orig(d_list: List[torch.Tensor],
             phi: torch.Tensor,
             horizon: int,
             n_states: int,
             n_actions: int,
             T: int,
             lambda_reg) -> torch.Tensor:
    """
    Original K-way objective function
    Computes -log det of information matrix with all pairwise interactions
    
    Args:
        phi: Feature matrix, either:
            - shape (n_states, feature_dim) for state-only features
            - shape (n_states * n_actions, feature_dim) for state-action features
    """
    is_state_only = phi.shape[0] == n_states
    
    # Process distributions based on feature type
    if is_state_only:
        d_processed = [d.view(horizon, n_states, n_actions).sum(dim=2) for d in d_list]
        feature_dim = n_states
    else:
        d_processed = [d.view(horizon, n_states * n_actions) for d in d_list]
        feature_dim = n_states * n_actions

    sigma_inv = torch.zeros((len(phi.T), len(phi.T)))
    
    for h in range(horizon):
        # Get all distributions at horizon h
        d_h = [d[h] for d in d_processed]
        
        # Sum all distributions for diagonal term
        diag_term = torch.diag(sum(d_h))
        
        # Sum over all pairs for outer products
        outer_sum = torch.zeros((feature_dim, feature_dim))
        for i in range(len(d_h)):
            for j in range(len(d_h)):
                if i != j:  # Skip self-pairs
                    outer_sum += d_h[i][:, None] @ d_h[j][:, None].T
        
        sigma_inv += T * phi.T @ (diag_term - outer_sum) @ phi

    sigma_inv += lambda_reg * torch.eye(len(phi.T))
    return -torch.logdet(sigma_inv)

def func_orig_grad(d_list: List[torch.Tensor],
                  phi: torch.Tensor,
                  horizon: int,
                  n_states: int,
                  n_actions: int,
                  T: int,
                  lambda_reg) -> List[torch.Tensor]:
    """
    Gradient of original K-way objective function
    Returns gradients for each distribution
    
    Args:
        phi: Feature matrix, either:
            - shape (n_states, feature_dim) for state-only features
            - shape (n_states * n_actions, feature_dim) for state-action features
    """
    is_state_only = phi.shape[0] == n_states

    # Process distributions based on feature type
    if is_state_only:
        d_processed = [d.view(horizon, n_states, n_actions) for d in d_list]
        d_sums = [d.sum(dim=2) for d in d_processed]
        feature_dim = n_states
    else:
        d_processed = [d.view(horizon, n_states * n_actions) for d in d_list]
        d_sums = d_processed  # no summing needed
        feature_dim = n_states * n_actions

    # Compute sigma_inv
    sigma_inv = torch.zeros((phi.shape[1], phi.shape[1]), device=d_processed[0].device)
    for h in range(horizon):
        d_h = [d[h] for d in d_sums]
        diag_term = torch.diag(sum(d_h))
        
        outer_sum = torch.zeros((feature_dim, feature_dim), device=d_processed[0].device)
        for i in range(len(d_h)):
            for j in range(len(d_h)):
                if i != j:
                    outer_sum += d_h[i][:, None] @ d_h[j][:, None].T
        
        sigma_inv += T * phi.T @ (diag_term - outer_sum) @ phi
    
    sigma_inv += lambda_reg * torch.eye(phi.shape[1], device=sigma_inv.device)
    M = torch.inverse(sigma_inv)
    phi_M_phi_T = phi @ M @ phi.T
    
    # Compute gradient for each distribution
    grad_list = []
    for k in range(len(d_list)):
        if is_state_only:
            grad_d = torch.zeros_like(d_processed[k])
            for h in range(horizon):
                # Diagonal term contribution
                grad_d[h] -= T * phi_M_phi_T.diag().unsqueeze(1)
                
                # Outer product terms contribution
                outer_sum = torch.zeros(n_states, device=d_processed[0].device)
                for j in range(len(d_list)):
                    if j != k:
                        outer_sum += d_sums[j][h]
                
                grad_d[h] += T * (phi_M_phi_T @ outer_sum + phi_M_phi_T.T @ outer_sum).unsqueeze(1)
        else:
            grad_d = torch.zeros_like(d_processed[k])
            for h in range(horizon):
                grad_d[h] -= T * phi_M_phi_T.diag()
                
                outer_sum = torch.zeros(n_states * n_actions, device=d_processed[0].device)
                for j in range(len(d_list)):
                    if j != k:
                        outer_sum += d_sums[j][h]
                
                grad_d[h] += T * (phi_M_phi_T @ outer_sum + phi_M_phi_T.T @ outer_sum)

        grad_list.append(grad_d.reshape(-1))
    
    return grad_list

# Aggregate objective function and gradient
def func_agg(d_list: List[torch.Tensor],
            phi: torch.Tensor,
            horizon: int,
            n_states: int,
            n_actions: int,
            T: int,
            lambda_reg) -> torch.Tensor:
    """
    Aggregate K-way objective function
    Uses single outer product of summed distributions
    
    Args:
        phi: Feature matrix, either:
            - shape (n_states, feature_dim) for state-only features
            - shape (n_states * n_actions, feature_dim) for state-action features
    """
    is_state_only = phi.shape[0] == n_states
    
    # Process distributions based on feature type
    if is_state_only:
        d_processed = [d.view(horizon, n_states, n_actions).sum(dim=2) for d in d_list]
        feature_dim = n_states
    else:
        d_processed = [d.view(horizon, n_states * n_actions) for d in d_list]
        feature_dim = n_states * n_actions

    sigma_inv = torch.zeros((phi.shape[1], phi.shape[1]), device=d_processed[0].device)

    for h in range(horizon):
        # Sum all distributions at horizon h
        d_total = sum(d[h] for d in d_processed)
        
        # Diagonal term
        term_diag = torch.diag(d_total)
        
        # Single outer product of the sum
        term_outer = 0.5 * torch.ger(d_total, d_total)
        
        sigma_inv += T * phi.T @ (term_diag - term_outer) @ phi

    sigma_inv += lambda_reg * torch.eye(phi.shape[1], device=sigma_inv.device)
    return -torch.logdet(sigma_inv)

def func_agg_grad(d_list: List[torch.Tensor],
                 phi: torch.Tensor,
                 horizon: int,
                 n_states: int,
                 n_actions: int,
                 T: int,
                 lambda_reg) -> List[torch.Tensor]:
    """
    Gradient of aggregate K-way objective function
    
    Args:
        phi: Feature matrix, either:
            - shape (n_states, feature_dim) for state-only features
            - shape (n_states * n_actions, feature_dim) for state-action features
    """
    is_state_only = phi.shape[0] == n_states
    
    # Process distributions based on feature type
    if is_state_only:
        d_processed = [d.view(horizon, n_states, n_actions) for d in d_list]
        d_sums = [d.sum(dim=2) for d in d_processed]
        feature_dim = n_states
    else:
        d_processed = [d.view(horizon, n_states * n_actions) for d in d_list]
        d_sums = d_processed
        feature_dim = n_states * n_actions
    
    # Total sum for each horizon
    d_total = sum(d_sums)
    
    sigma_inv = torch.zeros((phi.shape[1], phi.shape[1]), device=d_processed[0].device)
    for h in range(horizon):
        term_diag = torch.diag(d_total[h])
        term_outer = 0.5 * torch.ger(d_total[h], d_total[h])
        sigma_inv += T * phi.T @ (term_diag - term_outer) @ phi
    
    sigma_inv += lambda_reg * torch.eye(phi.shape[1], device=sigma_inv.device)
    M = torch.inverse(sigma_inv)
    A = phi @ M @ phi.T
    a_diag = A.diagonal()
    
    # Compute gradient for each distribution
    grad_list = []
    for k in range(len(d_list)):
        grad_d = torch.zeros_like(d_processed[k])
        
        for h in range(horizon):
            A_v = torch.matmul(d_total[h], A.T)
            if is_state_only:
                # Shape: [1, n_states, 1]
                a_diag_expanded = a_diag.unsqueeze(0).unsqueeze(-1)
                A_v_expanded = A_v.unsqueeze(-1)  # Shape: [n_states, 1]
                grad_d[h] = -T * (a_diag_expanded - A_v_expanded).expand(-1, -1, n_actions)
            else:
                grad_d[h] = -T * (a_diag - A_v)
        
        grad_list.append(grad_d.reshape(-1))
    
    return grad_list

def func_naive(d_list: List[torch.Tensor],
             phi: torch.Tensor,
             horizon: int,
             n_states: int,
             n_actions: int,
             T: int,
             lambda_reg) -> torch.Tensor:
   """
   Naive K-way objective function
   Only uses diagonal terms with summed distributions
   
   Args:
       phi: Feature matrix, either:
           - shape (n_states, feature_dim) for state-only features
           - shape (n_states * n_actions, feature_dim) for state-action features
   """
   is_state_only = phi.shape[0] == n_states
   
   # Process distributions based on feature type
   if is_state_only:
       d_processed = [d.view(horizon, n_states, n_actions).sum(dim=2) for d in d_list]
       feature_dim = n_states
   else:
       d_processed = [d.view(horizon, n_states * n_actions) for d in d_list]
       feature_dim = n_states * n_actions

   sigma_inv = torch.zeros((len(phi.T), len(phi.T)))

   for h in range(horizon):
       # Sum all distributions at horizon h
       d_total = sum(d[h] for d in d_processed)
       
       # Just use diagonal term
       sigma_inv += T * phi.T @ torch.diag(d_total) @ phi

   sigma_inv += lambda_reg * torch.eye(len(phi.T))
   return -torch.logdet(sigma_inv)

def func_naive_grad(d_list: List[torch.Tensor],
                  phi: torch.Tensor,
                  horizon: int,
                  n_states: int,
                  n_actions: int,
                  T: int,
                  lambda_reg) -> List[torch.Tensor]:
   """
   Gradient of naive K-way objective function
   
   Args:
       phi: Feature matrix, either:
           - shape (n_states, feature_dim) for state-only features
           - shape (n_states * n_actions, feature_dim) for state-action features
   """
   is_state_only = phi.shape[0] == n_states
   
   # Process distributions based on feature type
   if is_state_only:
       d_processed = [d.view(horizon, n_states, n_actions) for d in d_list]
       d_sums = [d.sum(dim=2) for d in d_processed]
       feature_dim = n_states
   else:
       d_processed = [d.view(horizon, n_states * n_actions) for d in d_list]
       d_sums = d_processed
       feature_dim = n_states * n_actions
   
   # Total sum over all distributions
   d_total = sum(d_sums)
   
   sigma_inv = T * sum(phi.T @ torch.diag(d_total[h]) @ phi for h in range(horizon))
   sigma_inv += lambda_reg * torch.eye(phi.shape[1], device=d_processed[0].device)
   
   M = torch.inverse(sigma_inv)
   phi_M_phi_T = phi @ M @ phi.T
   
   # Compute gradient for each distribution
   grad_list = []
   for k in range(len(d_list)):
       grad_d = torch.zeros_like(d_processed[k])
       
       for h in range(horizon):
           if is_state_only:
               # Shape: [1, n_states, 1]
               diag_term = phi_M_phi_T.diag().unsqueeze(0).unsqueeze(-1)
               grad_d[h] = -T * diag_term.expand(-1, -1, n_actions)
           else:
               grad_d[h] = -T * phi_M_phi_T.diag()
       
       grad_list.append(grad_d.reshape(-1))
   
   return grad_list

