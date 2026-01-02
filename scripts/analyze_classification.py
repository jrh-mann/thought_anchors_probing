#!/usr/bin/env python3
"""
Classification Probing Analysis Script

Trains logistic regression probes on activations to classify reward hacking.
Uses binary labels (threshold > 0 for counterfactual, outcome-based for baseline).
Primary metric: AUROC.

This enables fair comparison between:
- Counterfactual method: continuous labels binarized with threshold > 0
- Baseline method: binary outcome labels (smeared across all tokens)

Usage:
    # Single dataset
    python scripts/analyze_classification.py --base-dir /workspace/activations
    
    # Compare two methods
    python scripts/analyze_classification.py \
        --method1-dir /workspace/activations --method1-name "Counterfactual" \
        --method2-dir /workspace/activations2 --method2-name "Baseline"
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
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, roc_curve, confusion_matrix
)

# Import quick_load directly to avoid nnsight dependency
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
OUTPUT_DIR = "plots"
RESULTS_DIR = "results"
TEST_RATIO = 0.2
RANDOM_SEED = 42
BINARIZE_THRESHOLD = 0.0  # Labels > threshold become class 1
VAL_RATIO = 0.1  # Fraction of prompt-types reserved for validation (from non-negative types)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess(
    base_dir: str, 
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load activations and labels.
    
    Returns:
        X: (n_samples, n_layers, d_model) array
        y: (n_samples,) labels (continuous)
        prompt_names: List of prompt names
    """
    if verbose:
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        print(f"  Directory: {base_dir}")
    
    activations, metadata, labels = quick_load(base_dir, verbose=verbose)
    
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


def binarize_labels(
    y: np.ndarray, 
    threshold: float = 0.0
) -> np.ndarray:
    """
    Convert continuous labels to binary: label > threshold → class 1.
    
    For counterfactual method: any p_reward_hacks > 0 means hacking was detected.
    For baseline method: labels are already 0 or 1.
    """
    return (y > threshold).astype(int)


# ============================================================================
# TRAIN/TEST SPLIT BY PROMPT
# ============================================================================

def get_prompt_type(prompt_name: str) -> str:
    """Extract prompt type from prompt name."""
    # Baseline samples produced by scripts/extract_baseline_activations.py look like:
    #   baseline_{rollout_file.stem}_s{sentence_index}_r{rollout_idx}
    # where rollout_file.stem often ends with "_rollouts".
    if prompt_name.startswith("baseline_"):
        s = prompt_name[len("baseline_"):]
        # Strip trailing _s{...}_r{...}
        s = s.split("_s")[0] if "_s" in s else s
        if s.endswith("_rollouts"):
            s = s[: -len("_rollouts")]
        # Now apply the same prompt-type extraction as counterfactuals
        if "_rollout_" in s:
            return s.split("_rollout_")[0]
        return s
    
    if '_rollout_' in prompt_name:
        return prompt_name.split('_rollout_')[0]
    return prompt_name


def is_baseline_prompt(prompt_type: str) -> bool:
    """Identify baseline prompts."""
    return prompt_type.startswith("baseline_")


def is_negative_prompt(prompt_type: str) -> bool:
    """Identify negative sample prompts."""
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
    """
    np.random.seed(random_seed)
    
    # Group by prompt type
    prompt_types = list(set(get_prompt_type(p) for p in prompt_names))
    prompt_types.sort()
    
    # Filter out negative samples from test
    negative_types = [pt for pt in prompt_types if is_negative_prompt(pt)]
    other_types = [pt for pt in prompt_types if not is_negative_prompt(pt)]
    
    # Shuffle and split
    np.random.shuffle(other_types)
    n_test = max(1, int(len(other_types) * test_ratio))
    test_types = set(other_types[:n_test])
    train_types = set(other_types[n_test:]) | set(negative_types)
    
    print(f"\nPrompt type split:")
    print(f"  Train types ({len(train_types)}): {sorted(list(train_types))[:5]}...")
    print(f"  Test types ({len(test_types)}): {sorted(list(test_types))[:5]}...")
    
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
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    List[str], List[str], List[str]
]:
    """
    Three-way split by prompt type:
    - negative_* prompt types are forced into TRAIN only
    - remaining prompt types are split into TRAIN / VAL / TEST
    
    This avoids picking the "best layer" on the test set.
    """
    rng = np.random.RandomState(random_seed)
    
    prompt_types = sorted(set(get_prompt_type(p) for p in prompt_names))
    negative_types = sorted([pt for pt in prompt_types if is_negative_prompt(pt)])
    other_types = sorted([pt for pt in prompt_types if not is_negative_prompt(pt)])
    
    rng.shuffle(other_types)
    n_other = len(other_types)
    
    if n_other == 0:
        # Degenerate: only negatives exist
        train_types = set(prompt_types)
        val_types = set()
        test_types = set()
    else:
        n_test = max(1, int(round(n_other * test_ratio)))
        n_val = max(1, int(round(n_other * val_ratio))) if n_other >= 3 else 0
        
        test_types = set(other_types[:n_test])
        val_types = set(other_types[n_test:n_test + n_val])
        train_types = set(other_types[n_test + n_val:]) | set(negative_types)
        
        # If we accidentally emptied train (tiny n_other), fall back to at least one train type.
        if not train_types:
            train_types = set(val_types)
            val_types = set()
    
    print("\nPrompt type split (3-way):")
    print(f"  Train types ({len(train_types)}): {sorted(list(train_types))[:5]}...")
    print(f"  Val types   ({len(val_types)}): {sorted(list(val_types))[:5]}...")
    print(f"  Test types  ({len(test_types)}): {sorted(list(test_types))[:5]}...")
    if negative_types:
        print(f"  Negative-only train types ({len(negative_types)}): {negative_types[:5]}...")
    train_mask = np.array([get_prompt_type(p) in train_types for p in prompt_names])
    val_mask = np.array([get_prompt_type(p) in val_types for p in prompt_names])
    test_mask = np.array([get_prompt_type(p) in test_types for p in prompt_names])
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    train_prompts = [p for p, m in zip(prompt_names, train_mask) if m]
    val_prompts = [p for p, m in zip(prompt_names, val_mask) if m]
    test_prompts = [p for p, m in zip(prompt_names, test_mask) if m]
    
    print("\nSplit sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    return X_train, y_train, X_val, y_val, X_test, y_test, train_prompts, val_prompts, test_prompts


# ============================================================================
# LOGISTIC REGRESSION PROBE
# ============================================================================

def train_logistic_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train logistic regression and return classification metrics.
    
    Returns:
        Dictionary with:
        - auroc: Area under ROC curve
        - accuracy: Classification accuracy
        - precision: Precision for class 1
        - recall: Recall for class 1
        - f1: F1 score for class 1
        - pred_proba: Predicted probabilities for test set
        - model: Trained model
        - scaler: Fitted scaler
    """
    # Check for class balance
    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    n_pos_test = y_test.sum()
    n_neg_test = len(y_test) - n_pos_test
    
    if verbose:
        print(f"  Train: {n_pos_train:.0f} pos, {n_neg_train:.0f} neg")
        print(f"  Test: {n_pos_test:.0f} pos, {n_neg_test:.0f} neg")
    
    # Handle edge cases
    if n_pos_train == 0 or n_neg_train == 0:
        print("  Warning: Only one class in training data!")
        return {
            'auroc': 0.5, 'accuracy': 0.0, 'precision': 0.0, 
            'recall': 0.0, 'f1': 0.0, 'pred_proba': np.zeros(len(y_test)),
            'model': None, 'scaler': None
        }
    
    if n_pos_test == 0 or n_neg_test == 0:
        print("  Warning: Only one class in test data!")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(
        C=0.1,
        max_iter=100,
        solver='lbfgs',
        class_weight='balanced',  # Handle class imbalance
        random_state=RANDOM_SEED
    )
    model.fit(X_train_s, y_train)
    
    # Predictions
    pred = model.predict(X_test_s)
    pred_proba = model.predict_proba(X_test_s)[:, 1]
    
    # Metrics
    try:
        auroc = roc_auc_score(y_test, pred_proba)
    except ValueError:
        auroc = 0.5  # Only one class in test set
    
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred_proba': pred_proba,
        'model': model,
        'scaler': scaler
    }


def fit_logistic_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit a standardized logistic regression probe on (X_train, y_train).
    Returns a dict with keys: model, scaler.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=100,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_s, y_train)

    return {"model": model, "scaler": scaler}


def eval_logistic_probe(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate a fitted logistic probe on a dataset and return metrics.
    """
    X_s = scaler.transform(X)
    pred = model.predict(X_s)
    pred_proba = model.predict_proba(X_s)[:, 1]

    try:
        auroc = roc_auc_score(y, pred_proba)
    except ValueError:
        auroc = 0.5

    return {
        "auroc": float(auroc),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pred_proba": pred_proba,
    }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_dataset(
    base_dir: str,
    output_dir: str,
    method_name: str = "Method",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full classification analysis on a single dataset.
    
    Returns dictionary of results by layer.
    """
    print("\n" + "=" * 70)
    print(f"ANALYZING: {method_name}")
    print("=" * 70)
    print(f"  Directory: {base_dir}")
    
    # Load data
    X, y_continuous, prompt_names = load_and_preprocess(base_dir, verbose=verbose)
    n_samples, n_layers, d_model = X.shape
    
    # Binarize labels
    y = binarize_labels(y_continuous, BINARIZE_THRESHOLD)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    
    print(f"\nBinarized labels (threshold={BINARIZE_THRESHOLD}):")
    print(f"  Class 0 (clean): {n_neg} ({100*n_neg/len(y):.1f}%)")
    print(f"  Class 1 (hacking): {n_pos} ({100*n_pos/len(y):.1f}%)")
    
    # Split by prompt type (train/val/test)
    X_train, y_train_cont, X_val, y_val_cont, X_test, y_test_cont, *_ = split_by_prompt_train_val_test(
        X, y_continuous, prompt_names,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED,
    )
    
    # Binarize labels
    y_train = binarize_labels(y_train_cont, BINARIZE_THRESHOLD)
    y_val = binarize_labels(y_val_cont, BINARIZE_THRESHOLD) if len(y_val_cont) else np.array([], dtype=int)
    y_test = binarize_labels(y_test_cont, BINARIZE_THRESHOLD) if len(y_test_cont) else np.array([], dtype=int)
    
    print(f"\nTrain labels: {y_train.sum():.0f} pos, {len(y_train) - y_train.sum():.0f} neg")
    if len(y_val):
        print(f"Val labels:   {y_val.sum():.0f} pos, {len(y_val) - y_val.sum():.0f} neg")
    print(f"Test labels:  {y_test.sum():.0f} pos, {len(y_test) - y_test.sum():.0f} neg")
    
    # Train probes layer by layer
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC PROBES")
    print("=" * 60)
    
    results_by_layer = {}
    best_layer = None
    best_val_auroc = 0.0
    
    for layer_idx in range(n_layers):
        X_train_layer = X_train[:, layer_idx, :]
        X_val_layer = X_val[:, layer_idx, :] if len(X_val) else None
        X_test_layer = X_test[:, layer_idx, :] if len(X_test) else None

        # Train once; evaluate on train/val/test with the same fitted model.
        # Layer selection is done on VAL AUROC; TEST is held out for final reporting.
        fit = fit_logistic_probe(X_train_layer, y_train)
        model = fit["model"]
        scaler = fit["scaler"]

        result_train = eval_logistic_probe(model, scaler, X_train_layer, y_train)
        result_val = eval_logistic_probe(
            model,
            scaler,
            X_val_layer if X_val_layer is not None else X_train_layer,
            y_val if len(y_val) else y_train,
        )
        result_test = eval_logistic_probe(
            model,
            scaler,
            X_test_layer if X_test_layer is not None else X_train_layer,
            y_test if len(y_test) else y_train,
        )

        results_by_layer[layer_idx] = {
            "train": result_train,
            "val": result_val,
            "test": result_test,
        }
        
        print(
            f"  Layer {layer_idx:2d}: "
            f"Train AUROC={result_train['auroc']:.4f}, "
            f"Val AUROC={result_val['auroc']:.4f}, Test AUROC={result_test['auroc']:.4f}, "
            f"Test Acc={result_test['accuracy']:.4f}, Test F1={result_test['f1']:.4f}"
        )
        
        if result_val['auroc'] > best_val_auroc:
            best_val_auroc = result_val['auroc']
            best_layer = layer_idx
    
    best_test_auroc = results_by_layer[best_layer]["test"]["auroc"] if best_layer is not None else 0.0
    print(f"\n  Best layer (by VAL AUROC): {best_layer} (Val AUROC={best_val_auroc:.4f}, Test AUROC={best_test_auroc:.4f})")
    
    # Store summary
    results = {
        'method_name': method_name,
        'n_samples': n_samples,
        'n_layers': n_layers,
        'd_model': d_model,
        'n_train': len(y_train),
        'n_val': int(len(y_val)),
        'n_test': len(y_test),
        'class_balance': {
            'train_pos': int(y_train.sum()),
            'train_neg': int(len(y_train) - y_train.sum()),
            'val_pos': int(y_val.sum()) if len(y_val) else 0,
            'val_neg': int(len(y_val) - y_val.sum()) if len(y_val) else 0,
            'test_pos': int(y_test.sum()),
            'test_neg': int(len(y_test) - y_test.sum())
        },
        'best_layer': best_layer,
        'best_val_auroc': best_val_auroc,
        'best_test_auroc': best_test_auroc,
        'by_layer': {
            layer: {
                'val': {k: v for k, v in res['val'].items() if k not in ['model', 'scaler', 'pred_proba']},
                'test': {k: v for k, v in res['test'].items() if k not in ['model', 'scaler', 'pred_proba']},
            }
            for layer, res in results_by_layer.items()
        },
        'y_test': y_test,
        'best_pred_proba': results_by_layer[best_layer]['test']['pred_proba'] if best_layer is not None else np.array([])
    }
    
    return results


def plot_auroc_by_layer(
    results: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None
) -> None:
    """Plot AUROC by layer."""
    layers = sorted(results['by_layer'].keys())
    # Plot TEST AUROC by default (what we care about after val-based selection)
    aurocs = [results['by_layer'][l]['test']['auroc'] for l in layers]
    
    plt.figure(figsize=(12, 5))
    plt.bar(layers, aurocs, color='steelblue', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
    
    best_layer = results['best_layer']
    plt.bar(best_layer, aurocs[best_layer], color='darkgreen', alpha=0.9, label=f'Best: {best_layer}')
    
    plt.xlabel('Layer')
    plt.ylabel('AUROC')
    plt.title(title or f"{results['method_name']}: Test AUROC by Layer")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    pred_proba: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
    label: str = "Model"
) -> None:
    """Plot ROC curve for best layer."""
    fpr, tpr, _ = roc_curve(y_true, pred_proba)
    auroc = roc_auc_score(y_true, pred_proba)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'{label} (AUROC={auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title or 'ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_method_comparison(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    output_path: str
) -> None:
    """Plot AUROC comparison between two methods."""
    layers = sorted(results1['by_layer'].keys())
    aurocs1 = [results1['by_layer'][l]['test']['auroc'] for l in layers]
    aurocs2 = [results2['by_layer'][l]['test']['auroc'] for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    plt.figure(figsize=(14, 6))
    bars1 = plt.bar(x - width/2, aurocs1, width, label=results1['method_name'], alpha=0.8)
    bars2 = plt.bar(x + width/2, aurocs2, width, label=results2['method_name'], alpha=0.8)
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    
    plt.xlabel('Layer')
    plt.ylabel('AUROC')
    plt.title('AUROC Comparison by Layer')
    plt.xticks(x, layers)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_roc_comparison(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    output_path: str
) -> None:
    """Plot ROC curves for both methods on same axes."""
    plt.figure(figsize=(8, 8))
    
    for results, color in [(results1, 'blue'), (results2, 'orange')]:
        y_test = results['y_test']
        pred_proba = results['best_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        auroc = roc_auc_score(y_test, pred_proba)
        
        plt.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{results['method_name']} (AUROC={auroc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Best Layers)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Classification probing analysis")
    
    # Single dataset mode
    parser.add_argument("--base-dir", type=str, help="Directory with activations (single mode)")
    parser.add_argument("--method-name", type=str, default="Method", help="Name for single method")
    
    # Comparison mode
    parser.add_argument("--method1-dir", type=str, help="Directory for method 1")
    parser.add_argument("--method1-name", type=str, default="Counterfactual", help="Name for method 1")
    parser.add_argument("--method2-dir", type=str, help="Directory for method 2")
    parser.add_argument("--method2-name", type=str, default="Baseline", help="Name for method 2")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="Output directory for JSON")
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine mode
    comparison_mode = args.method1_dir and args.method2_dir
    single_mode = args.base_dir
    
    if not comparison_mode and not single_mode:
        print("Error: Specify either --base-dir (single mode) or --method1-dir and --method2-dir (comparison mode)")
        return 1
    
    if comparison_mode:
        # ================================================================
        # COMPARISON MODE
        # ================================================================
        print("\n" + "=" * 70)
        print("CLASSIFICATION COMPARISON ANALYSIS")
        print("=" * 70)
        print(f"\nMethod 1: {args.method1_name} ({args.method1_dir})")
        print(f"Method 2: {args.method2_name} ({args.method2_dir})")
        
        # Analyze both datasets
        results1 = analyze_dataset(
            args.method1_dir, args.output_dir, args.method1_name
        )
        results2 = analyze_dataset(
            args.method2_dir, args.output_dir, args.method2_name
        )
        
        # Generate comparison plots
        print("\n" + "=" * 60)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 60)
        
        plot_auroc_by_layer(
            results1, 
            str(Path(args.output_dir) / f"auroc_by_layer_{args.method1_name.lower()}.png"),
            f"{args.method1_name}: AUROC by Layer"
        )
        plot_auroc_by_layer(
            results2,
            str(Path(args.output_dir) / f"auroc_by_layer_{args.method2_name.lower()}.png"),
            f"{args.method2_name}: AUROC by Layer"
        )
        
        plot_method_comparison(
            results1, results2,
            str(Path(args.output_dir) / "method_comparison.png")
        )
        
        plot_roc_comparison(
            results1, results2,
            str(Path(args.output_dir) / "roc_comparison.png")
        )
        
        # Save results
        comparison_results = {
            args.method1_name: {
                k: v for k, v in results1.items() 
                if k not in ['y_test', 'best_pred_proba']
            },
            args.method2_name: {
                k: v for k, v in results2.items()
                if k not in ['y_test', 'best_pred_proba']
            },
            'comparison': {
                'method1_best_test_auroc': results1['best_test_auroc'],
                'method2_best_test_auroc': results2['best_test_auroc'],
                'difference': results1['best_test_auroc'] - results2['best_test_auroc'],
                'method1_best_layer': results1['best_layer'],
                'method2_best_layer': results2['best_layer']
            }
        }
        
        results_path = Path(args.results_dir) / "classification_comparison.json"
        with open(results_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"  Saved: {results_path}")
        
        # Summary
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n{args.method1_name}:")
        print(f"  Best layer: {results1['best_layer']}")
        print(f"  Best Val AUROC: {results1['best_val_auroc']:.4f}")
        print(f"  Best Test AUROC: {results1['best_test_auroc']:.4f}")
        
        print(f"\n{args.method2_name}:")
        print(f"  Best layer: {results2['best_layer']}")
        print(f"  Best Val AUROC: {results2['best_val_auroc']:.4f}")
        print(f"  Best Test AUROC: {results2['best_test_auroc']:.4f}")
        
        diff = results1['best_test_auroc'] - results2['best_test_auroc']
        winner = args.method1_name if diff > 0 else args.method2_name
        print(f"\nDifference: {abs(diff):.4f} (winner: {winner})")
        
    else:
        # ================================================================
        # SINGLE DATASET MODE
        # ================================================================
        results = analyze_dataset(
            args.base_dir, args.output_dir, args.method_name
        )
        
        # Generate plots
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        plot_auroc_by_layer(
            results,
            str(Path(args.output_dir) / "auroc_by_layer.png")
        )
        
        plot_roc_curve(
            results['y_test'],
            results['best_pred_proba'],
            str(Path(args.output_dir) / "roc_curve_best.png"),
            f"ROC Curve - Layer {results['best_layer']}"
        )
        
        # Save results
        results_out = {
            k: v for k, v in results.items()
            if k not in ['y_test', 'best_pred_proba']
        }
        results_path = Path(args.results_dir) / "classification_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results_out, f, indent=2)
        print(f"  Saved: {results_path}")
        
        # Summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"\nDataset: {args.base_dir}")
        print(f"Samples: {results['n_samples']}")
        print(f"Best layer: {results['best_layer']}")
        print(f"Best Val AUROC: {results['best_val_auroc']:.4f}")
        print(f"Best Test AUROC: {results['best_test_auroc']:.4f}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

