"""
Example usage of CoIF for fusing zero-cost proxies.
"""

import numpy as np
import pickle
import os
from scipy.stats import spearmanr
from coif import CoIF


def load_data(data_dir: str, dataset: str = 'cifar10'):
    """
    Load proxy data from files.
    
    Args:
        data_dir: Directory containing data files
        dataset: Dataset name ('cifar10', 'cifar100', 'imagenet')
    
    Returns:
        Tuple of (accuracy, proxy_dict)
    """
    acc = np.load(os.path.join(data_dir, f'{dataset}_acc.npy'), allow_pickle=True)
    
    with open(os.path.join(data_dir, f'{dataset}_proxy_dict.pkl'), 'rb') as f:
        proxy_dict = pickle.load(f)
    
    return acc, proxy_dict


def main():
    # Example with synthetic data
    print("=" * 60)
    print("CoIF Example with Synthetic Data")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 10000
    n_proxies = 10
    
    # Generate synthetic target (e.g., accuracy)
    target = np.random.randn(n_samples)
    
    # Generate synthetic proxies with different correlations to target
    proxy_values = []
    for i in range(n_proxies):
        noise = np.random.randn(n_samples) * (1 + min(0.3*i, 5) * 0.3)
        proxy = target + noise
        proxy_values.append(proxy)
    proxy_values = np.array(proxy_values)
    
    # Print individual proxy correlations
    print("\nIndividual proxy correlations with target:")
    for i in range(n_proxies):
        corr, _ = spearmanr(proxy_values[i], target)
        print(f"  Proxy {i+1}: {corr:.4f}")
    
    # Use CoIF to fuse proxies
    print("\n" + "-" * 40)
    print("Fitting CoIF model...")
    
    coif = CoIF(use_rank=True)
    combined = coif.fit_transform(proxy_values, target)
    
    result, _ = spearmanr(combined, target)
    print(f"\nCoIF result:")
    print(f"  Fused correlation: {result:.4f}")
    print(f"  Weights (normalized): {coif.get_weights()}")
    print(f"  Weights sum: {np.sum(coif.get_weights()):.4f}")
    
    
    # Subsampling example
    print("\n" + "-" * 40)
    print("Subsampling example (fit on 20% data, evaluate on all):")
    
    sample_size = int(n_samples * 0.2)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    
    coif_sub = CoIF(use_rank=True)
    coif_sub.fit(proxy_values, target, sample_indices=sample_indices)
    
    # Evaluate on all data
    eval_corr = coif_sub.evaluate(proxy_values, target)
    print(f" Test Correlation: {eval_corr:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
