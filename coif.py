"""
CoIF: Covariance-Inverse Fusion for Zero-Cost Proxies

This module provides a closed-form solution for fusing multiple zero-cost proxies
to achieve better correlation with the true performance metric.
"""

import numpy as np
from scipy.stats import rankdata, spearmanr


def compute_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute optimal weights using covariance matrix.
    
    The weights are computed as: w = C^(-1) @ ones / sqrt(ones @ C^(-1) @ ones)
    
    Args:
        cov_matrix: Covariance matrix of proxy values (N x N)
    
    Returns:
        Optimal weight vector (N,)
    """
    if cov_matrix.ndim != 2:
        raise ValueError("Covariance matrix must be 2-dimensional")
    
    n_rows, n_cols = cov_matrix.shape
    if n_rows != n_cols:
        raise ValueError("Covariance matrix must be square")
    
    N = n_rows
    ones = np.ones(N)
    
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular, using pseudo-inverse")
        cov_inv = np.linalg.pinv(cov_matrix)
    
    numerator = cov_inv @ ones
    denominator = np.sqrt(ones @ cov_inv @ ones)
    weights = numerator / denominator
    
    return weights


def optimize_weights(rho: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Compute optimal weights using correlation-based closed-form solution.
    
    Objective: maximize w^T @ rho / sqrt(w^T @ C @ w)
    Closed-form solution: w = C^(-1) @ rho
    
    Args:
        rho: Correlation vector between each proxy and target (N,)
        C: Correlation matrix between proxies (N x N)
    
    Returns:
        Optimal weight vector (N,)
    """
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C)
    
    w = C_inv @ rho
    return w


def process_proxy_values(proxy_values: np.ndarray, 
                         use_rank: bool = True, 
                         normalize: bool = False) -> np.ndarray:
    """
    Process proxy values by handling NaN/Inf and optionally applying rank transform.
    
    Args:
        proxy_values: Raw proxy values (n_samples,)
        use_rank: If True, apply rank transformation
        normalize: If True and use_rank is False, normalize to [0, 1]
    
    Returns:
        Processed proxy values (n_samples,)
    """
    proxy_values = np.array(proxy_values).flatten()
    
    # Handle NaN and Inf
    proxy_values[np.isnan(proxy_values)] = 0
    proxy_values[np.isinf(proxy_values)] = 0
    
    if use_rank:
        return rankdata(proxy_values, method='min')
    elif normalize:
        min_val, max_val = np.min(proxy_values), np.max(proxy_values)
        if max_val - min_val > 0:
            return (proxy_values - min_val) / (max_val - min_val)
    
    return proxy_values


def fuse_proxies(proxy_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Fuse multiple proxy values using given weights.
    
    Args:
        proxy_values: Proxy values matrix (n_proxies, n_samples)
        weights: Weight vector (n_proxies,)
    
    Returns:
        Fused proxy values (n_samples,)
    """
    proxy_values = np.array(proxy_values)
    return np.dot(weights, proxy_values)


class CoIF:
    """
    CoIF: Covariance-Inverse Fusion for Zero-Cost Proxies.
    
    This class provides methods to fuse multiple zero-cost proxies
    using a closed-form optimization approach.
    
    Example:
        >>> coif = CoIF(use_rank=True)
        >>> coif.fit(proxy_values, target_values)
        >>> combined = coif.transform(proxy_values)
        >>> # Or use fit_transform
        >>> combined = coif.fit_transform(proxy_values, target_values)
    """
    
    def __init__(self, use_rank: bool = True, normalize: bool = False, 
                 use_rank_target: bool = True):
        """
        Initialize CoIF.
        
        Args:
            use_rank: If True, apply rank transformation to proxy values
            normalize: If True and use_rank is False, normalize values to [0, 1]
            use_rank_target: If True, use spearmanr (rank-based) for correlation with target;
                            If False, use pearsonr (no rank) for correlation with target
        """
        self.use_rank = use_rank
        self.normalize = normalize
        self.use_rank_target = use_rank_target
        self.weights_ = None
        self.n_proxies_ = None
    
    def fit(self, proxy_values: np.ndarray, target_values: np.ndarray, 
            sample_indices: np.ndarray = None) -> 'CoIF':
        """
        Fit the CoIF model to compute optimal weights.
        
        Args:
            proxy_values: Proxy values matrix (n_proxies, n_samples)
            target_values: Target values (n_samples,), e.g., accuracy
            sample_indices: Optional indices for subsampling
        
        Returns:
            self
        """
        proxy_values = np.array(proxy_values)
        target_values = np.array(target_values)
        
        if proxy_values.ndim == 1:
            proxy_values = proxy_values.reshape(1, -1)
        
        self.n_proxies_ = proxy_values.shape[0]
        
        # Process each proxy
        processed_proxies = np.array([
            process_proxy_values(pv, self.use_rank, self.normalize) 
            for pv in proxy_values
        ])
        
        # Subsample if indices provided
        if sample_indices is not None:
            proxy_subset = processed_proxies[:, sample_indices]
            target_subset = target_values[sample_indices]
        else:
            proxy_subset = processed_proxies
            target_subset = target_values
        
        # Compute correlation between each proxy and target
        if self.use_rank_target:
            # Use Spearman correlation (rank-based)
            rho = np.array([
                spearmanr(proxy_subset[i], target_subset)[0] 
                for i in range(self.n_proxies_)
            ])
        else:
            # Use Pearson correlation (no rank transformation on target)
            from scipy.stats import pearsonr
            rho = np.array([
                pearsonr(proxy_subset[i], target_subset)[0] 
                for i in range(self.n_proxies_)
            ])
        
        # Compute covariance matrix between proxies
        cov_matrix = np.cov(proxy_subset)
        
        # Compute optimal weights
        weights = optimize_weights(rho, cov_matrix)
        
        # Normalize weights to sum to 1
        self.weights_ = weights / np.sum(np.abs(weights))
        
        return self
    
    def transform(self, proxy_values: np.ndarray) -> np.ndarray:
        """
        Transform proxy values using fitted weights.
        
        Args:
            proxy_values: Proxy values matrix (n_proxies, n_samples)
        
        Returns:
            Fused proxy values (n_samples,)
        """
        if self.weights_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        proxy_values = np.array(proxy_values)
        
        if proxy_values.ndim == 1:
            proxy_values = proxy_values.reshape(1, -1)
        
        # Process each proxy
        processed_proxies = np.array([
            process_proxy_values(pv, self.use_rank, self.normalize) 
            for pv in proxy_values
        ])
        
        return fuse_proxies(processed_proxies, self.weights_)
    
    def fit_transform(self, proxy_values: np.ndarray, target_values: np.ndarray,
                      sample_indices: np.ndarray = None) -> np.ndarray:
        """
        Fit the model and transform proxy values.
        
        Args:
            proxy_values: Proxy values matrix (n_proxies, n_samples)
            target_values: Target values (n_samples,)
            sample_indices: Optional indices for subsampling during fitting
        
        Returns:
            Fused proxy values (n_samples,)
        """
        self.fit(proxy_values, target_values, sample_indices)
        return self.transform(proxy_values)
    
    def get_weights(self) -> np.ndarray:
        """
        Get the fitted weights.
        
        Returns:
            Weight vector (n_proxies,)
        """
        if self.weights_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.weights_.copy()
    
    def evaluate(self, proxy_values: np.ndarray, target_values: np.ndarray) -> float:
        """
        Evaluate the combined proxy correlation with target.
        
        Args:
            proxy_values: Proxy values matrix (n_proxies, n_samples)
            target_values: Target values (n_samples,)
        
        Returns:
            Spearman correlation coefficient
        """
        combined = self.transform(proxy_values)
        correlation, _ = spearmanr(combined, target_values)
        return correlation
