from typing import List, Tuple, Optional
import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

def llm_mdp(
    list_of_text_tokens: List[List[str]], 
    horizon: int,
    device: str = 'cpu',
    models_cache_dir: str = '/tmp/models_cache_dir/',
    beta: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    assert horizon == len(list_of_text_tokens)+1
    
    # Get unique tokens
    unique_tokens = list(set(token for tokens in list_of_text_tokens for token in tokens))
    token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
    n_actions = len(unique_tokens)
    
    # State space: start state + token states + terminal state
    n_states = 1 + n_actions + 1
    START_STATE = 0
    TERMINAL_STATE = n_states - 1
    
    # Initialize CLIP
    clip_model_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_id, cache_dir=models_cache_dir)
    model = CLIPModel.from_pretrained(clip_model_id, cache_dir=models_cache_dir).to(device)
    
    # Generate CLIP embeddings for tokens
    @torch.no_grad()
    def get_clip_embedding(text: str) -> torch.Tensor:
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        return model.get_text_features(**inputs).detach().cpu().double()
    
    # Compute and store token embeddings
    token_embeddings = {token: get_clip_embedding(token) for token in unique_tokens}
    feature_dim = next(iter(token_embeddings.values())).shape[1]
    
    # Construct phi matrix for all state-action pairs
    phi = torch.zeros((n_states * n_actions, feature_dim))
    
    # 1. Start state features: β * CLIP(a)
    for token, a in token_to_idx.items():
        idx = START_STATE * n_actions + a
        phi[idx] = beta * token_embeddings[token]


    # 2. Token state features: CLIP(s) + β * CLIP(a)
    for s_token, s_idx in token_to_idx.items():
        state_idx = s_idx + 1  # +1 for start state offset
        for a_token, a in token_to_idx.items():
            idx = state_idx * n_actions + a
            phi[idx] = token_embeddings[s_token] + beta * token_embeddings[a_token]
    
    # 3. Terminal state features: CLIP(s)
    for token, a in token_to_idx.items():
        idx = TERMINAL_STATE * n_actions + a
        phi[idx] = token_embeddings[token]
    
    # Build transition matrix (horizon, n_states, n_actions, n_states)
    transitions = torch.zeros((horizon, n_states, n_actions, n_states))

    # this should never happen since we start in the START_STATE
    transitions[0, 1:, :, TERMINAL_STATE] = 1.0    
    transitions[1:, START_STATE, :, TERMINAL_STATE] = 1.0    
    
    # At h=0: can only transition from start state to token states
    for token, action_idx in token_to_idx.items():
        state_idx = token_to_idx[token] + 1  # +1 for start state offset
        transitions[0, START_STATE, action_idx, state_idx] = 1.0
    
    # Middle horizons: can transition between token states
    for s in range(1, TERMINAL_STATE):  # For each state
        for token, action_idx in token_to_idx.items():  # For each action
            state_idx = token_to_idx[token] + 1  # Where this action leads
            transitions[1:horizon-2, s, action_idx, state_idx] = 1.0
    

    # At final horizon: must transition to terminal state
    transitions[horizon-2, 1:TERMINAL_STATE, :, TERMINAL_STATE] = 1.0

    # Terminal state is absorbing 
    transitions[:, TERMINAL_STATE, :, TERMINAL_STATE] = 1.0
    transitions[horizon-1, :, :, TERMINAL_STATE] = 1.0

    # Sanity checks for transitions matrix (horizon, n_states, n_actions, n_states)

    # 1. Check probabilities sum to 1 for each state-action pair at each horizon
    assert torch.allclose(transitions.sum(dim=-1), torch.ones_like(transitions.sum(dim=-1))), \
        "Transition probabilities must sum to 1 for each state-action pair"

    # 2. Check start state transitions (h=0)
    assert torch.all(transitions[0, START_STATE, :, 1:TERMINAL_STATE].sum() > 0), \
        "Start state should have transitions to token states at h=0"
    assert torch.all(transitions[0, START_STATE, :, [START_STATE, TERMINAL_STATE]] == 0), \
        "Start state shouldn't transition to itself or terminal at h=0"

    # 3. Check terminal state is absorbing at last horizon
    assert torch.all(transitions[horizon-1, TERMINAL_STATE, :, TERMINAL_STATE] == 1.0), \
        "Terminal state must be absorbing at final horizon"

    # 4. Check horizon-2 transitions go to terminal
    assert torch.all(transitions[horizon-2, 1:TERMINAL_STATE, :, TERMINAL_STATE] == 1.0), \
        "All states must transition to terminal state at horizon-2"
    
    # Initial state distribution
    init_state_dist = torch.zeros(n_states)
    init_state_dist[START_STATE] = 1.0
    
    # Set theta as the CLIP embedding of "Cheese"
    theta = get_clip_embedding("jump").squeeze().float()
    theta += beta*get_clip_embedding("dog").squeeze().float()
    
    return transitions, theta, init_state_dist, phi

def llm_simple_mdp(
    token_file: str,
    horizon: int,
    seed: int,
    device: str = 'cpu',
    models_cache_dir: str = '/tmp/models_cache_dir/',
    pca_dim: int = 50
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    cache_file = f"/tmp/llm_token_embeddings_{os.path.basename(token_file)}_{pca_dim}.npz"
    
    if seed != 1 and not os.path.exists(cache_file):
        print(f"Process {seed} waiting for token embeddings cache...")
        return None

    # Load tokens
    with open(token_file, 'r') as file:
        unique_tokens = sorted(set(line.strip() for line in file))
    n_actions = len(unique_tokens)
    n_states = horizon

    if os.path.exists(cache_file):
        cache = np.load(cache_file, allow_pickle=True)
        token_embeddings = cache['token_embeddings'].item()
        theta = torch.from_numpy(cache['theta'])
    else:
        print(f"Process {seed} computing token embeddings...")
        clip_model_id = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(clip_model_id, cache_dir=models_cache_dir)
        model = CLIPModel.from_pretrained(clip_model_id, cache_dir=models_cache_dir).to(device)
        
        @torch.no_grad()
        def get_clip_embedding(text: str) -> torch.Tensor:
            inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, 
                             truncation=True, return_tensors="pt").to(device)
            return model.get_text_features(**inputs).detach().cpu().double()
        
        # Get theta and token embeddings
        theta = get_clip_embedding("jump").squeeze().float()
        embeddings_list = [get_clip_embedding(token) for token in unique_tokens]
        token_embeddings_matrix = torch.stack(embeddings_list).squeeze(1)
        
        # Center embeddings and apply PCA if needed
        embeddings_mean = token_embeddings_matrix.mean(dim=0, keepdim=True)
        centered_embeddings = token_embeddings_matrix - embeddings_mean
        #theta = theta - embeddings_mean.squeeze()
        theta = theta.double()
        
        if pca_dim != 768:
            U, S, V = torch.svd(centered_embeddings)
            token_embeddings_matrix = centered_embeddings @ V[:, :pca_dim]
            theta = V[:, :pca_dim].T @ theta
            

        # Create dictionary from reduced embeddings
        token_embeddings = {token: emb for token, emb in zip(unique_tokens, token_embeddings_matrix)}
        
        print(f"Process {seed} saving token embeddings to {cache_file}")
        np.savez(cache_file, token_embeddings=token_embeddings, theta=theta.numpy())
        return None

    # Rest of the function remains the same
    feature_dim = next(iter(token_embeddings.values())).shape[0]
    embeddings_list = [token_embeddings[token] for token in unique_tokens]
    token_embeddings_matrix = torch.stack(embeddings_list)
    
    # Create phi matrix
    phi = token_embeddings_matrix.repeat(n_states, 1)
    
    # Build transition matrix efficiently
    transitions = torch.zeros((horizon, n_states, n_actions, n_states))
    # Set next state transitions for all but last state/horizon
    row_indices = torch.arange(n_states-1)
    transitions[:-1, row_indices, :, row_indices+1] = 1.0
    # Last state and final horizon transitions
    transitions[:, -1, :, -1] = 1.0
    transitions[-1, :, :, -1] = 1.0
    # Initial state distribution
    init_state_dist = torch.zeros(n_states)
    init_state_dist[0] = 1.0
    
    return transitions, theta, init_state_dist, phi

