# CoIF
CoIF is a method for fusing multiple zero-cost proxies to achieve better correlation with the true performance metric in Neural Architecture Search (NAS).

## Installation

```bash
pip install numpy scipy
```

## Quick Start

```python
import numpy as np
from coif import CoIF

# Prepare your proxy values (n_proxies, n_samples)
proxy_values = np.array([
    [...],  # proxy 1 values
    [...],  # proxy 2 values
    [...],  # proxy 3 values
])

# Target values (e.g., accuracy)
target_values = np.array([...])

# Create and fit CoIF model
coif = CoIF(use_rank=True)
coif.fit(proxy_values, target_values)

# Get fused proxy values
fused = coif.transform(proxy_values)

# Or use fit_transform
fused = coif.fit_transform(proxy_values, target_values)

# Evaluate correlation
correlation = coif.evaluate(proxy_values, target_values)
print(f"Fused correlation: {correlation:.4f}")

# Get learned weights
weights = coif.get_weights()
```

## Example

See `example.py` for a complete working example.

```bash
python example.py
```
## API Reference

### `CoIF.fit(proxy_values, target_values, sample_indices=None)`

Fit the model to compute optimal weights.

- `proxy_values`: Proxy values matrix (n_proxies, n_samples)
- `target_values`: Target values (n_samples,)
- `sample_indices`: Optional indices for subsampling during training

### `CoIF.transform(proxy_values)`

Transform proxy values using fitted weights.

### `CoIF.fit_transform(proxy_values, target_values, sample_indices=None)`

Fit and transform in one step.

### `CoIF.evaluate(proxy_values, target_values)`

Evaluate Spearman correlation between combined proxy and target.

### `CoIF.get_weights()`

Get the fitted weight vector.

## Subsampling

For large search spaces, you can fit on a subset of architectures:

```python
# Randomly sample 200 architectures for training
sample_indices = np.random.choice(n_samples, 200, replace=False)

coif = CoIF()
coif.fit(proxy_values, target_values, sample_indices=sample_indices)

# Evaluate on all architectures
fused = coif.transform(proxy_values)
```



## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{coif2025,
  title={Covariance-Inverse Fusion: Towards Comprehensive Proxies for Zero-Shot Neural Architecture Search},
  author={Wu, Ning and Huang, Han and Kang, Li and Xu, Yueting and Feng, Fujian and Wu, Chunguo},
  journal={},
  year={2025}
}
```

## License

MIT License
