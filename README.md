# Tauformer

**A Transformer architecture using taumode attention for memory-efficient sequence modeling**

Tauformer replaces standard dot-product attention with **taumode distance scoring**, compressing token representations into scalar synthetic indices derived from feature-space topological analysis. This enables constant-memory key storage while maintaining full attention expressiveness.

## Overview

Standard transformer attention computes \( O(T^2) \) pairwise scores via \( QK^\top \), then applies softmax and weighted sum with \( V \). Tauformer innovates by:

- **Scalar indexing**: Each query/key is compressed into a scalar \( \lambda \) via Rayleigh quotient energy on a feature-space graph Laplacian (see `arrowspace-rs` paper).
- **Distance-based scoring**: Attention scores use \( -|\lambda_q - \lambda_k| / T \) instead of dot products
- **Memory-first caching**: Stores only \( (\lambda_k, V) \) tuples, not full \( K \) vectors

This provides significant memory savings for long contexts while preserving the causal masking and softmax stability pipeline.

## Architecture

### taumode Attention

The core innovation replaces the scoring mechanism:

**Standard attention:**
```
scores = Q @ K.T # (B, H, Tq, Tk) matrix multiply
att = softmax(causal_mask(scores))
out = att @ V
```


**taumode attention:**
```
lambda_q = taumode(Q, L) # (B, H, Tq) scalars
lambda_k = taumode(K, L) # (B, H, Tk) scalars
scores = -|lambda_q[:,:,:,None] - lambda_k[:,:,None,:]| / temperature
att = softmax(causal_mask(scores))
out = att @ V
```


Where `L` is a **feature-space Laplacian** (F×F matrix, F = head dimension) built from a domain corpus using `ArrowSpace`.

### Synthetic Lambda Computation

The `taumode(x, L)` function computes a bounded Rayleigh quotient [file:10]:
```
E_raw = (x^T L x) / (x^T x + eps)
E_bounded = E_raw / (E_raw + tau)
lambda = E_bounded # synthetic spectral index
```

This measures the "spectral roughness" of vector `x` with respect to the feature manifold encoded in `L`.
