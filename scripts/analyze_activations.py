#!/usr/bin/env python3
"""
Activation Probing Analysis Script

Trains linear and MLP probes on activations to predict reward hacking probability.
Generates visualizations and saves metrics.

Usage:
    python scripts/analyze_activations.py
    python scripts/analyze_activations.py --base-dir workspace/activations --output-dir plots
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import quick_load directly to avoid nnsight dependency in __init__.py
import importlib.util
_load_spec = importlib.util.spec_from_file_location(
    "load", 
    str(Path(__file__).parent.parent / "src" / "activations" / "load.py")
)
_load_module = importlib.util.module_from_spec(_load_spec)
_load_spec.loader.exec_module(_load_module)
quick_load = _load_module.quick_load

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = "/workspace/activations"
OUTPUT_DIR = "plots"
RESULTS_DIR = "results"
MLP_HIDDEN_SIZES = [2]
TEST_RATIO = 0.2
VAL_RATIO = 0.1
MLP_EPOCHS = 100  # Fewer epochs with early stopping
MLP_LR = 0.001     # Higher LR with scheduler
MLP_BATCH_SIZE = 1024
RANDOM_SEED = 42

# Regularization defaults (can be overridden via CLI args)
RIDGE_ALPHA = 1.0
MLP_WEIGHT_DECAY = 1e-4
MLP_DROPOUT = 0.0

# ============================================================================
# METRICS HELPERS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics plus a diagnostic R² on non-zero labels.
    
    Why include a non-zero diagnostic?
    If the label distribution is heavily zero-inflated (e.g., many sentences have
    p_reward_hacks == 0), overall R² can be dominated by the model learning to
    predict ~0 everywhere. The non-zero R² is not "more correct", but it is a
    useful sanity check for whether we're learning signal beyond the mass at 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Diagnostic: R² computed only on non-zero labels
    mask_nz = y_true > 0.0
    if mask_nz.any():
        r2_nz = float(r2_score(y_true[mask_nz], y_pred[mask_nz]))
    else:
        r2_nz = float("nan")
    
    return {"mse": mse, "r2": r2, "r2_nonzero": r2_nz}


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize inputs using TRAIN statistics only.
    
    Returns:
        X_train_s, X_test_s, mean, std
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + eps
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def save_mlp_2layer_artifact(
    out_path: Path,
    *,
    layer_idx: int,
    hidden_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    model_state_dict: Dict[str, torch.Tensor],
    ridge_alpha: float,
    mlp_weight_decay: float,
    mlp_dropout: float,
    huber_beta: float,
    notes: str = ""
) -> None:
    """
    Save a production-friendly artifact for the *2-layer* probe as-is, plus its
    standardization stats.
    
    In production:
      x_s = (x - mean) / std
      y_hat = MLPProbe(...).load_state_dict(model_state_dict)(x_s)
    
    Saved format is a torch checkpoint for easy loading.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "layer_idx": int(layer_idx),
        "hidden_size": int(hidden_size),
        "mean": mean.reshape(-1).astype(np.float32),
        "std": std.reshape(-1).astype(np.float32),
        "model_state_dict": {k: v.detach().cpu() for k, v in model_state_dict.items()},
        "train_hparams": {
            "ridge_alpha": float(ridge_alpha),
            "mlp_weight_decay": float(mlp_weight_decay),
            "mlp_dropout": float(mlp_dropout),
            "huber_beta": float(huber_beta),
        },
        "notes": notes,
    }
    torch.save(ckpt, out_path)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess(base_dir: str, verbose: bool = True, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load activations (already flattened to token-samples by quick_load).
    
    Args:
        base_dir: Directory containing activation files
        verbose: Print progress messages
        max_samples: Limit number of samples for testing
        
    Returns:
        X: (n_samples, n_layers, d_model) array
        y: (n_samples,) labels
        prompt_names: List of prompt names
    """
    if verbose:
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        if max_samples:
            print(f"  (Limited to first {max_samples} samples for testing)")
    
    # Load activations - quick_load now flattens tokens to samples automatically
    activations, metadata, labels = quick_load(base_dir, verbose=verbose)
    
    # Limit samples if requested
    if max_samples and len(activations) > max_samples:
        activations = activations[:max_samples]
        metadata = metadata[:max_samples]
        labels = labels[:max_samples]
        if verbose:
            print(f"  Limited to {max_samples} samples")
    
    # Data is already (n_samples, n_layers, d_model)
    X = activations
    y = labels
    prompt_names = [m['prompt_name'] for m in metadata]
    
    if verbose:
        print(f"\nData shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique prompts: {len(set(prompt_names))}")
        print(f"Label range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"Label mean: {y.mean():.3f}")
    
    return X, y, prompt_names


# ============================================================================
# TRAIN/TEST SPLIT BY PROMPT
# ============================================================================

def get_prompt_type(prompt_name: str) -> str:
    """
    Extract prompt type from prompt name.
    E.g., 'audio_mfcc_rollout_0' -> 'audio_mfcc'
    """
    # Split on '_rollout_' and take first part
    if '_rollout_' in prompt_name:
        return prompt_name.split('_rollout_')[0]
    return prompt_name


def is_negative_prompt(prompt_type: str) -> bool:
    """
    Identify negative-sample prompts produced by scripts/extract_negative_activations.py.
    
    These are created under directory names like 'negative_0', 'negative_1', ...
    We generally do NOT want these to appear in the test split, because they are
    distributionally different from the counterfactual rollouts and can make
    evaluation artificially easy via dataset-ID shortcuts.
    """
    return prompt_type.startswith("negative_")


def split_by_prompt(
    X: np.ndarray,
    y: np.ndarray,
    prompt_names: List[str],
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Split data ensuring entire prompt types are held out.
    
    Returns:
        X_train, y_train, X_test, y_test, train_prompts, test_prompts
    """
    np.random.seed(random_seed)
    
    # Group by prompt type
    prompt_types = list(set(get_prompt_type(p) for p in prompt_names))
    prompt_types.sort()
    
    # Ensure negative samples never end up in the test split
    negative_types = sorted([pt for pt in prompt_types if is_negative_prompt(pt)])
    non_negative_types = sorted([pt for pt in prompt_types if not is_negative_prompt(pt)])
    
    # Shuffle and split *non-negative* prompt types
    np.random.shuffle(non_negative_types)
    if len(non_negative_types) == 0:
        # Degenerate case: only negative prompts exist. We'll keep everything in train.
        test_types = set()
        train_types = set(prompt_types)
    else:
        n_test = max(1, int(len(non_negative_types) * test_ratio))
        test_types = set(non_negative_types[:n_test])
        train_types = set(non_negative_types[n_test:]) | set(negative_types)
    
    print(f"\nPrompt type split:")
    print(f"  Train types ({len(train_types)}): {sorted(train_types)}")
    print(f"  Test types ({len(test_types)}): {sorted(test_types)}")
    if negative_types:
        print(f"  Negative-only train types ({len(negative_types)}): {negative_types}")
    
    # Create masks
    train_mask = np.array([get_prompt_type(p) in train_types for p in prompt_names])
    test_mask = np.array([get_prompt_type(p) in test_types for p in prompt_names])
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    train_prompts = [p for p, m in zip(prompt_names, train_mask) if m]
    test_prompts = [p for p, m in zip(prompt_names, test_mask) if m]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, y_train, X_test, y_test, train_prompts, test_prompts


def split_by_prompt_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    prompt_names: List[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    List[str], List[str], List[str]
]:
    """
    3-way split by prompt type:
    - negative_* are forced into TRAIN only
    - remaining prompt types split into TRAIN/VAL/TEST
    """
    rng = np.random.RandomState(random_seed)
    
    prompt_types = sorted(set(get_prompt_type(p) for p in prompt_names))
    negative_types = sorted([pt for pt in prompt_types if is_negative_prompt(pt)])
    non_negative_types = sorted([pt for pt in prompt_types if not is_negative_prompt(pt)])
    
    rng.shuffle(non_negative_types)
    n = len(non_negative_types)
    
    if n == 0:
        train_types, val_types, test_types = set(prompt_types), set(), set()
    else:
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        test_types = set(non_negative_types[:n_test])
        val_types = set(non_negative_types[n_test:n_test + n_val])
        train_types = set(non_negative_types[n_test + n_val:]) | set(negative_types)
        if not train_types:
            train_types = set(val_types)
            val_types = set()
    
    print(f"\nPrompt type split (3-way):")
    print(f"  Train types ({len(train_types)}): {sorted(train_types)}")
    print(f"  Val types   ({len(val_types)}): {sorted(val_types)}")
    print(f"  Test types  ({len(test_types)}): {sorted(test_types)}")
    if negative_types:
        print(f"  Negative-only train types ({len(negative_types)}): {negative_types}")
    
    train_mask = np.array([get_prompt_type(p) in train_types for p in prompt_names])
    val_mask = np.array([get_prompt_type(p) in val_types for p in prompt_names])
    test_mask = np.array([get_prompt_type(p) in test_types for p in prompt_names])
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    train_prompts = [p for p, m in zip(prompt_names, train_mask) if m]
    val_prompts = [p for p, m in zip(prompt_names, val_mask) if m]
    test_prompts = [p for p, m in zip(prompt_names, test_mask) if m]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_prompts, val_prompts, test_prompts


# ============================================================================
# LINEAR PROBE
# ============================================================================

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0
) -> Dict[str, float]:
    """
    Train Ridge regression (L2-regularized linear) and return metrics.
    
    Uses Ridge instead of LinearRegression because:
    - Much faster for high-dimensional data (closed-form solution is more stable)
    - Small regularization prevents numerical issues
    - Nearly identical results for small alpha
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        alpha: Ridge regularization strength (default 1.0)
        
    Returns:
        Dict with train_mse, test_mse, train_r2, test_r2
    """
    # Standardize features (important for Ridge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    return {
        'train_mse': float(mean_squared_error(y_train, y_train_pred)),
        'test_mse': float(mean_squared_error(y_test, y_test_pred)),
        'train_r2': float(r2_score(y_train, y_train_pred)),
        'test_r2': float(r2_score(y_test, y_test_pred)),
        'train_r2_nonzero': compute_metrics(y_train, y_train_pred)['r2_nonzero'],
        'test_r2_nonzero': compute_metrics(y_test, y_test_pred)['r2_nonzero'],
        'y_test_pred': y_test_pred,
        'y_test_true': y_test
    }


# ============================================================================
# MLP PROBE
# ============================================================================

class MLPProbe(nn.Module):
    """
    "MLP" probe.
    
    NOTE: By user request, this probe is *linear* (no nonlinearity). With two
    Linear layers and no activation, the entire network is equivalent to a
    single Linear layer (with bias). We keep the hidden layer structure only
    so you can vary hidden_size and keep the surrounding code unchanged.
    """
    
    def __init__(self, input_dim: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # Intentionally linear: no activation.
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_size: int,
    epochs: int = 500,
    lr: float = 0.01,
    batch_size: int = 1024,
    patience: int = 50,
    weight_decay: float = 1e-4,
    dropout: float = 0.0,
    huber_beta: float = 1.0,
    log_every: int = 10,
    verbose: bool = True,
    print_prefix: str = ""
) -> Dict[str, float]:
    """
    Train MLP probe with mini-batch SGD, early stopping, and LR scheduling.
    
    Optimizations:
    - Mini-batch training (faster per epoch, better gradients)
    - Early stopping (stop when no improvement)
    - Learning rate scheduling (reduce on plateau)
    - Proper initialization (Xavier)
    - Input standardization (zero mean, unit variance)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Standardize inputs using TRAIN statistics (production requires saving these).
    X_train_s, X_val_s, X_mean, X_std = standardize_train_test(X_train, X_val)
    X_test_s = (X_test - X_mean) / X_std
    
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
    
    # Create dataset and dataloader for mini-batch training
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = MLPProbe(X_train.shape[1], hidden_size, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=huber_beta)
    
    # Early stopping
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        
        epoch_loss /= len(X_train_t)
        
        # Evaluate on VAL (for scheduling + early stopping)
        model.eval()
        with torch.no_grad():
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_loss = float(criterion(model(X_val_t), y_val_t).item())
        
        scheduler.step(val_loss)
        
        # Optional diagnostic: evaluate on TEST to spot overtraining (do not early-stop on it)
        test_loss = None
        with torch.no_grad():
            y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
            test_loss = float(criterion(model(X_test_t), y_test_t).item())
        
        # Progress logging
        if log_every is not None and log_every > 0:
            if epoch == 0 or (epoch + 1) % log_every == 0:
                if test_loss is None:
                    print(f"{print_prefix}Epoch {epoch+1:4d}/{epochs} | train_loss={epoch_loss:.6f} | val_loss={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.6f}")
                else:
                    print(f"{print_prefix}Epoch {epoch+1:4d}/{epochs} | train_loss={epoch_loss:.6f} | val_loss={val_loss:.6f} | test_loss={test_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"{print_prefix}Early stop at epoch {epoch+1} (no val loss improvement for {patience} epochs)")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_t).cpu().numpy()
        y_val_pred = model(X_val_t).cpu().numpy()
        y_test_pred = model(X_test_t).cpu().numpy()
    
    return {
        'train_mse': float(mean_squared_error(y_train, y_train_pred)),
        'train_r2': float(r2_score(y_train, y_train_pred)),
        'train_r2_nonzero': compute_metrics(y_train, y_train_pred)['r2_nonzero'],
        'val_mse': float(mean_squared_error(y_val, y_val_pred)),
        'val_r2': float(r2_score(y_val, y_val_pred)),
        'val_r2_nonzero': compute_metrics(y_val, y_val_pred)['r2_nonzero'],
        'test_mse': float(mean_squared_error(y_test, y_test_pred)),
        'test_r2': float(r2_score(y_test, y_test_pred)),
        'test_r2_nonzero': compute_metrics(y_test, y_test_pred)['r2_nonzero'],
        'input_mean': X_mean.reshape(-1),
        'input_std': X_std.reshape(-1),
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'y_test_pred': y_test_pred,
        'y_test_true': y_test
    }


# ============================================================================
# PCA VISUALIZATION
# ============================================================================

def create_pca_html(
    X_by_layer: Dict[int, np.ndarray],
    y: np.ndarray,
    output_path: str,
    n_components: int = 3
) -> None:
    """
    Create interactive Plotly HTML with PCA for each layer.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: plotly not installed, skipping PCA HTML")
        return
    
    print("\nCreating PCA visualizations...")
    
    # Compute PCA for each layer
    pca_results = {}
    for layer_idx, X_layer in X_by_layer.items():
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_layer)
        pca_results[layer_idx] = {
            'coords': X_pca,
            'explained_variance': pca.explained_variance_ratio_
        }
    
    # Create figure with dropdown
    fig = go.Figure()
    
    layers = sorted(pca_results.keys())
    
    # Add trace for each layer (only first visible)
    for i, layer_idx in enumerate(layers):
        coords = pca_results[layer_idx]['coords']
        ev = pca_results[layer_idx]['explained_variance']
        
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=y,
                colorscale='RdYlGn_r',
                colorbar=dict(title='p_reward_hacks') if i == 0 else None,
                opacity=0.7
            ),
            name=f'Layer {layer_idx}',
            visible=(i == 0),
            hovertemplate=f'PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<br>Label: %{{marker.color:.2f}}<extra>Layer {layer_idx}<br>Var: {ev[0]:.1%}, {ev[1]:.1%}, {ev[2]:.1%}</extra>'
        ))
    
    # Create dropdown menu
    buttons = []
    for i, layer_idx in enumerate(layers):
        visibility = [False] * len(layers)
        visibility[i] = True
        ev = pca_results[layer_idx]['explained_variance']
        buttons.append(dict(
            label=f'Layer {layer_idx}',
            method='update',
            args=[{'visible': visibility},
                  {'title': f'PCA - Layer {layer_idx} (Explained var: {ev[0]:.1%}, {ev[1]:.1%}, {ev[2]:.1%})'}]
        ))
    
    fig.update_layout(
        title=f'PCA - Layer {layers[0]}',
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.1,
            y=1.15
        )],
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=100, b=0)
    )
    
    # Save
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_layer_performance(
    linear_results: Dict[int, Dict],
    mlp_results: Dict[int, Dict[int, Dict]],
    output_path: str
) -> None:
    """Plot MSE vs layer for linear and MLP probes."""
    
    layers = sorted(linear_results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE plot
    ax = axes[0]
    
    # Linear probe
    linear_test_mse = [linear_results[l]['test_mse'] for l in layers]
    ax.plot(layers, linear_test_mse, 'o-', label='Linear', linewidth=2, markersize=6)
    
    # MLP probes
    hidden_sizes = sorted(set(h for layer_results in mlp_results.values() for h in layer_results.keys()))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(hidden_sizes)))
    
    for hidden_size, color in zip(hidden_sizes, colors):
        mlp_test_mse = [mlp_results[l].get(hidden_size, {}).get('test_mse', np.nan) for l in layers]
        ax.plot(layers, mlp_test_mse, 's--', label=f'MLP(h={hidden_size})', 
                color=color, linewidth=1.5, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE by Layer')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # R² plot
    ax = axes[1]
    
    linear_test_r2 = [linear_results[l]['test_r2'] for l in layers]
    ax.plot(layers, linear_test_r2, 'o-', label='Linear', linewidth=2, markersize=6)
    
    for hidden_size, color in zip(hidden_sizes, colors):
        mlp_test_r2 = [mlp_results[l].get(hidden_size, {}).get('test_r2', np.nan) for l in layers]
        ax.plot(layers, mlp_test_r2, 's--', label=f'MLP(h={hidden_size})',
                color=color, linewidth=1.5, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Test R²')
    ax.set_title('Test R² by Layer')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: str
) -> None:
    """Scatter plot of predictions vs actual values."""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    # Stats
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    ax.set_xlabel('Actual p_reward_hacks')
    ax.set_ylabel('Predicted p_reward_hacks')
    ax.set_title(f'{title}\nMSE={mse:.4f}, R²={r2:.4f}, r={corr:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_mlp_comparison(
    mlp_results: Dict[int, Dict[int, Dict]],
    output_path: str
) -> None:
    """Heatmap of MLP performance: layers x hidden sizes."""
    
    layers = sorted(mlp_results.keys())
    hidden_sizes = sorted(set(h for layer_results in mlp_results.values() for h in layer_results.keys()))
    
    # Create matrices
    train_mse = np.zeros((len(layers), len(hidden_sizes)))
    test_mse = np.zeros((len(layers), len(hidden_sizes)))
    
    for i, layer in enumerate(layers):
        for j, h in enumerate(hidden_sizes):
            if h in mlp_results[layer]:
                train_mse[i, j] = mlp_results[layer][h]['train_mse']
                test_mse[i, j] = mlp_results[layer][h]['test_mse']
            else:
                train_mse[i, j] = np.nan
                test_mse[i, j] = np.nan
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Train MSE heatmap
    im1 = axes[0].imshow(train_mse, aspect='auto', cmap='viridis_r')
    axes[0].set_xticks(range(len(hidden_sizes)))
    axes[0].set_xticklabels(hidden_sizes)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers)
    axes[0].set_xlabel('Hidden Size')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Train MSE')
    plt.colorbar(im1, ax=axes[0])
    
    # Test MSE heatmap
    im2 = axes[1].imshow(test_mse, aspect='auto', cmap='viridis_r')
    axes[1].set_xticks(range(len(hidden_sizes)))
    axes[1].set_xticklabels(hidden_sizes)
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(layers)
    axes[1].set_xlabel('Hidden Size')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('Test MSE')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze activation probes")
    parser.add_argument("--base-dir", type=str, default=BASE_DIR, help="Activations directory")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="Output directory for metrics")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS, help="MLP training epochs")
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA, help="Ridge (linear probe) L2 regularization strength")
    parser.add_argument("--mlp-weight-decay", type=float, default=MLP_WEIGHT_DECAY, help="MLP AdamW weight decay (L2 regularization)")
    parser.add_argument("--mlp-dropout", type=float, default=MLP_DROPOUT, help="MLP dropout probability (0 disables)")
    parser.add_argument("--huber-beta", type=float, default=1.0, help="Huber beta for SmoothL1Loss (default: 1.0)")
    parser.add_argument("--save-probe-artifacts", action="store_true", default=True,
                       help="Save production-ready probe artifacts (default: True)")
    parser.add_argument("--no-save-probe-artifacts", action="store_true",
                       help="Disable saving probe artifacts")
    parser.add_argument("--skip-mlp", action="store_true", help="Skip MLP training (faster)")
    parser.add_argument("--skip-pca", action="store_true", help="Skip PCA visualization")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing (before token expansion)")
    parser.add_argument("--test-layers", type=str, default=None, 
                       help="Comma-separated layer indices to test (e.g., '0,12,23'). Default: all layers")
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # ========================================================================
    # 1. Load and preprocess data
    # ========================================================================
    X, y, prompt_names = load_and_preprocess(args.base_dir, max_samples=args.max_samples)
    n_samples, n_layers, d_model = X.shape
    
    # Parse test layers
    if args.test_layers:
        test_layer_indices = [int(x.strip()) for x in args.test_layers.split(',')]
        print(f"\nTesting only layers: {test_layer_indices}")
    else:
        test_layer_indices = list(range(n_layers))
    
    # ========================================================================
    # 2. Split by prompt type
    # ========================================================================
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, X_test, y_test, train_prompts, val_prompts, test_prompts = split_by_prompt_train_val_test(
        X, y, prompt_names, val_ratio=args.val_ratio, test_ratio=args.test_ratio, random_seed=RANDOM_SEED
    )
    
    # ========================================================================
    # 3. Train linear probes for each layer
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRAINING LINEAR PROBES")
    print("=" * 60)
    
    linear_results = {}
    best_layer_linear = None
    best_test_mse_linear = float('inf')
    
    for layer_idx in test_layer_indices:
        X_train_layer = X_train[:, layer_idx, :]
        X_test_layer = X_test[:, layer_idx, :]
        
        results = train_linear_probe(X_train_layer, y_train, X_test_layer, y_test, alpha=args.ridge_alpha)
        linear_results[layer_idx] = results
        
        if results['test_mse'] < best_test_mse_linear:
            best_test_mse_linear = results['test_mse']
            best_layer_linear = layer_idx
        
        print(
            f"  Layer {layer_idx:2d}: "
            f"Train MSE={results['train_mse']:.4f}, Train R²={results['train_r2']:.4f} (nz={results['train_r2_nonzero']:.4f}); "
            f"Test MSE={results['test_mse']:.4f}, Test R²={results['test_r2']:.4f} (nz={results['test_r2_nonzero']:.4f})"
        )
    
    print(f"\n  Best layer: {best_layer_linear} (Test MSE={best_test_mse_linear:.4f})")
    
    # ========================================================================
    # 4. Random label ablation
    # ========================================================================
    print("\n" + "=" * 60)
    print("RANDOM LABEL ABLATION")
    print("=" * 60)
    
    y_train_random = np.random.permutation(y_train)
    y_test_random = np.random.permutation(y_test)
    
    random_results = {}
    for layer_idx in test_layer_indices:
        X_train_layer = X_train[:, layer_idx, :]
        X_test_layer = X_test[:, layer_idx, :]
        
        results = train_linear_probe(X_train_layer, y_train_random, X_test_layer, y_test_random, alpha=args.ridge_alpha)
        random_results[layer_idx] = results
        
        print(
            f"  [random] Layer {layer_idx:2d}: "
            f"Train MSE={results['train_mse']:.4f}, Train R²={results['train_r2']:.4f} (nz={results['train_r2_nonzero']:.4f}); "
            f"Test MSE={results['test_mse']:.4f}, Test R²={results['test_r2']:.4f} (nz={results['test_r2_nonzero']:.4f})"
        )
    
    # Compare
    real_mean_r2 = np.mean([linear_results[l]['test_r2'] for l in test_layer_indices])
    random_mean_r2 = np.mean([random_results[l]['test_r2'] for l in test_layer_indices])
    
    print(f"  Real labels - Mean Test R²: {real_mean_r2:.4f}")
    print(f"  Random labels - Mean Test R²: {random_mean_r2:.4f}")
    print(f"  Difference: {real_mean_r2 - random_mean_r2:.4f}")
    
    # ========================================================================
    # 5. Train MLP probes for each layer and hidden size
    # ========================================================================
    mlp_results = defaultdict(dict)
    
    if not args.skip_mlp:
        print("\n" + "=" * 60)
        print("TRAINING MLP PROBES")
        print("=" * 60)
        
        for layer_idx in test_layer_indices:
            X_train_layer = X_train[:, layer_idx, :]
            X_val_layer = X_val[:, layer_idx, :]
            X_test_layer = X_test[:, layer_idx, :]
            
            print(f"\n  Layer {layer_idx}:")
            
            for hidden_size in MLP_HIDDEN_SIZES:
                results = train_mlp_probe(
                    X_train_layer, y_train, X_val_layer, y_val, X_test_layer, y_test,
                    hidden_size=hidden_size,
                    epochs=args.mlp_epochs,
                    lr=MLP_LR,
                    batch_size=MLP_BATCH_SIZE,
                    weight_decay=args.mlp_weight_decay,
                    dropout=args.mlp_dropout,
                    huber_beta=args.huber_beta,
                    log_every=10,
                    verbose=True,
                    print_prefix="      "
                )
                mlp_results[layer_idx][hidden_size] = results
                
                print(
                    f"    h={hidden_size:2d}: "
                    f"Train MSE={results['train_mse']:.4f}, Train R²={results['train_r2']:.4f} (nz={results['train_r2_nonzero']:.4f}); "
                    f"Val MSE={results['val_mse']:.4f}, Val R²={results['val_r2']:.4f} (nz={results['val_r2_nonzero']:.4f}); "
                    f"Test MSE={results['test_mse']:.4f}, Test R²={results['test_r2']:.4f} (nz={results['test_r2_nonzero']:.4f})"
                )
                
                # Save a production-friendly artifact (2-layer net + standardization)
                save_artifacts = args.save_probe_artifacts and not args.no_save_probe_artifacts
                if save_artifacts:
                    out_path = Path(args.results_dir) / "probe_artifacts" / f"mlp_linear_layer_{layer_idx}_h{hidden_size}.pt"
                    save_mlp_2layer_artifact(
                        out_path,
                        layer_idx=layer_idx,
                        hidden_size=hidden_size,
                        mean=results["input_mean"],
                        std=results["input_std"],
                        model_state_dict=results["model_state_dict"],
                        ridge_alpha=args.ridge_alpha,
                        mlp_weight_decay=args.mlp_weight_decay,
                        mlp_dropout=args.mlp_dropout,
                        huber_beta=args.huber_beta,
                        notes="Standardized 2-layer linear (no activation) probe saved as-is.",
                    )
    
    # ========================================================================
    # 6. PCA Visualization
    # ========================================================================
    if not args.skip_pca:
        print("\n" + "=" * 60)
        print("CREATING PCA VISUALIZATIONS")
        print("=" * 60)
        
        # Combine train and test for visualization
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        X_by_layer = {layer_idx: X_all[:, layer_idx, :] for layer_idx in test_layer_indices}
        
        create_pca_html(
            X_by_layer, y_all,
            output_path=str(Path(args.output_dir) / "pca_all_layers.html")
        )
    
    # ========================================================================
    # 7. Generate plots
    # ========================================================================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    # Layer performance plot
    plot_layer_performance(
        linear_results, dict(mlp_results),
        output_path=str(Path(args.output_dir) / "layer_performance.png")
    )
    
    # Pred vs actual for best layer (linear)
    best_results = linear_results[best_layer_linear]
    plot_pred_vs_actual(
        best_results['y_test_true'],
        best_results['y_test_pred'],
        title=f"Linear Probe - Layer {best_layer_linear}",
        output_path=str(Path(args.output_dir) / "pred_vs_actual_linear_best.png")
    )
    
    # MLP comparison heatmap
    if mlp_results:
        plot_mlp_comparison(
            dict(mlp_results),
            output_path=str(Path(args.output_dir) / "mlp_hidden_size_comparison.png")
        )
        
        # Pred vs actual for best MLP (selected by VAL, reported on TEST)
        best_mlp_layer = None
        best_mlp_h = None
        best_mlp_val_mse = float('inf')
        
        for layer_idx in mlp_results:
            for h in mlp_results[layer_idx]:
                mse = mlp_results[layer_idx][h]['val_mse']
                if mse < best_mlp_val_mse:
                    best_mlp_val_mse = mse
                    best_mlp_layer = layer_idx
                    best_mlp_h = h
        
        if best_mlp_layer is not None:
            best_mlp_results = mlp_results[best_mlp_layer][best_mlp_h]
            plot_pred_vs_actual(
                best_mlp_results['y_test_true'],
                best_mlp_results['y_test_pred'],
                title=f"MLP Probe (h={best_mlp_h}) - Layer {best_mlp_layer}",
                output_path=str(Path(args.output_dir) / "pred_vs_actual_mlp_best.png")
            )
            # Also save a canonical "best" artifact name for production convenience
            save_artifacts = args.save_probe_artifacts and not args.no_save_probe_artifacts
            if save_artifacts:
                out_path = Path(args.results_dir) / "probe_artifacts" / "mlp_linear_best.pt"
                save_mlp_2layer_artifact(
                    out_path,
                    layer_idx=int(best_mlp_layer),
                    hidden_size=int(best_mlp_h),
                    mean=best_mlp_results["input_mean"],
                    std=best_mlp_results["input_std"],
                    model_state_dict=best_mlp_results["model_state_dict"],
                    ridge_alpha=args.ridge_alpha,
                    mlp_weight_decay=args.mlp_weight_decay,
                    mlp_dropout=args.mlp_dropout,
                    huber_beta=args.huber_beta,
                    notes="Best standardized 2-layer probe selected by VAL MSE (test used only for reporting).",
                )
    
    # ========================================================================
    # 8. Save metrics
    # ========================================================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Prepare results for JSON (remove numpy arrays)
    def clean_for_json(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = clean_for_json(v)
            elif isinstance(v, np.ndarray):
                continue  # Skip arrays
            elif isinstance(v, torch.Tensor):
                continue  # Skip tensors (e.g., model_state_dict) in JSON outputs
            elif isinstance(v, (np.floating, np.integer)):
                result[k] = float(v)
            else:
                result[k] = v
        return result
    
    # Summary metrics
    metrics = {
        'n_samples': int(n_samples),
        'n_layers': int(n_layers),
        'd_model': int(d_model),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'regularization': {
            'ridge_alpha': float(args.ridge_alpha),
            'mlp_weight_decay': float(args.mlp_weight_decay),
            'mlp_dropout': float(args.mlp_dropout),
        },
        'best_linear_layer': int(best_layer_linear),
        'best_linear_test_mse': float(best_test_mse_linear),
        'best_linear_test_r2': float(linear_results[best_layer_linear]['test_r2']),
        'random_ablation': {
            'real_mean_r2': float(real_mean_r2),
            'random_mean_r2': float(random_mean_r2),
            'difference': float(real_mean_r2 - random_mean_r2)
        }
    }
    
    with open(Path(args.results_dir) / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {args.results_dir}/metrics.json")
    
    # Linear probe results by layer
    linear_clean = {str(k): clean_for_json(v) for k, v in linear_results.items()}
    with open(Path(args.results_dir) / "linear_probe_by_layer.json", 'w') as f:
        json.dump(linear_clean, f, indent=2)
    print(f"  Saved: {args.results_dir}/linear_probe_by_layer.json")
    
    # MLP results by layer and size
    if mlp_results:
        mlp_clean = {str(k): {str(h): clean_for_json(v) for h, v in layer_dict.items()} 
                     for k, layer_dict in mlp_results.items()}
        with open(Path(args.results_dir) / "mlp_probes_by_layer_and_size.json", 'w') as f:
            json.dump(mlp_clean, f, indent=2)
        print(f"  Saved: {args.results_dir}/mlp_probes_by_layer_and_size.json")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nPlots saved to: {args.output_dir}/")
    print(f"Metrics saved to: {args.results_dir}/")
    print(f"\nKey findings:")
    print(f"  - Best linear probe layer: {best_layer_linear} (R²={linear_results[best_layer_linear]['test_r2']:.4f})")
    print(f"  - Random label ablation: Real R² - Random R² = {real_mean_r2 - random_mean_r2:.4f}")
    if mlp_results and best_mlp_layer is not None:
        print(f"  - Best MLP: Layer {best_mlp_layer}, h={best_mlp_h} (R²={mlp_results[best_mlp_layer][best_mlp_h]['test_r2']:.4f})")


if __name__ == "__main__":
    main()

