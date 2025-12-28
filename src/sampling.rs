// src/sampling.rs

//! Sampling strategies for text generation
//!
//! Returns token IDs as [B, 1] shaped tensors for consistent batch handling

use burn::tensor::{Bool, Int, Tensor, TensorData, activation, backend::Backend};
use log::debug;

#[derive(Clone, Copy)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state.
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use top 24 bits -> [0,1)
        let v = (self.next_u64() >> 40) as u32; // 24 bits
        (v as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

#[inline]
fn sample_multinomial_row(probs: &[f32], rng: &mut XorShift64) -> usize {
    let mut r = rng.next_f32();
    let mut cum = 0.0f32;

    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i;
        }
    }
    // Numerical fallback
    probs.len().saturating_sub(1)
}

fn sample_from_probs<B: Backend>(probs: Tensor<B, 2>, rng: &mut XorShift64) -> Tensor<B, 2, Int> {
    let [b, v] = probs.dims();
    let host: Vec<f32> = probs.to_data().to_vec().unwrap();

    let mut out: Vec<i64> = Vec::with_capacity(b);
    for bi in 0..b {
        let row = &host[bi * v..(bi + 1) * v];
        out.push(sample_multinomial_row(row, rng) as i64);
    }

    Tensor::<B, 1, Int>::from_data(TensorData::new(out, [b]), &probs.device()).reshape([b, 1])
}

// ═════════════════════════════════════════════════════════════════════════════
// Temperature scaling
// ═════════════════════════════════════════════════════════════════════════════

pub fn apply_temperature<B: Backend>(logits: Tensor<B, 2>, temperature: f64) -> Tensor<B, 2> {
    if temperature == 1.0 {
        return logits;
    }
    assert!(
        temperature > 0.0,
        "Temperature must be positive, got {}",
        temperature
    );
    debug!("Applying temperature scaling: {}", temperature);
    logits / temperature
}

// ═════════════════════════════════════════════════════════════════════════════
// Top-k filtering
// ═════════════════════════════════════════════════════════════════════════════

pub fn top_k_filter<B: Backend>(logits: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
    let [batch, vocab] = logits.dims();
    if k == 0 || k >= vocab {
        return logits;
    }

    // CPU pass: compute kth-largest threshold per row.
    let host: Vec<f32> = logits.to_data().to_vec().unwrap();
    let mut keep_mask = vec![false; batch * vocab];

    for b in 0..batch {
        // Build (value, idx) pairs for this row
        let mut pairs: Vec<(f32, usize)> = (0..vocab).map(|v| (host[b * vocab + v], v)).collect();

        // Sort descending by value (largest first)
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // kth largest value is at index (k-1)
        let kth_val = pairs[k - 1].0;

        // Keep all entries >= kth_val (handles ties deterministically)
        for v in 0..vocab {
            keep_mask[b * vocab + v] = host[b * vocab + v] >= kth_val;
        }
    }

    let keep_data = TensorData::new(keep_mask, [batch, vocab]);
    let keep = Tensor::<B, 2, Bool>::from_data(keep_data, &logits.device());
    logits.mask_fill(keep.bool_not(), f64::NEG_INFINITY)
}

// ═════════════════════════════════════════════════════════════════════════════
// Top-p (nucleus) filtering
// ═════════════════════════════════════════════════════════════════════════════

pub fn top_p_filter<B: Backend>(logits: Tensor<B, 2>, p: f64) -> Tensor<B, 2> {
    assert!(p > 0.0 && p <= 1.0, "top_p must be in (0, 1]");
    let [batch, vocab] = logits.dims();

    if p >= 0.9999 {
        return logits;
    }

    debug!("Applying top-p filter: p={}", p);

    let probs = activation::softmax(logits.clone(), 1);
    let probs_host: Vec<f32> = probs.to_data().to_vec().unwrap();

    // Build keep mask as bools directly
    let mut keep_mask_bool: Vec<bool> = vec![false; batch * vocab];

    for b in 0..batch {
        let mut pairs: Vec<(f32, usize)> =
            (0..vocab).map(|v| (probs_host[b * vocab + v], v)).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut cum = 0.0f32;
        for (prob, idx) in pairs {
            if cum < p as f32 {
                keep_mask_bool[b * vocab + idx] = true;
                cum += prob;
            } else {
                break;
            }
        }
    }

    // Create bool mask directly without comparison
    let keep_mask_data = TensorData::new(keep_mask_bool, [batch, vocab]);
    let keep_bool = Tensor::<B, 2, Bool>::from_data(keep_mask_data, &probs.device());

    logits.mask_fill(keep_bool.bool_not(), f64::NEG_INFINITY)
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling functions - All return [B, 1] Int tensors
// ═════════════════════════════════════════════════════════════════════════════

/// Greedy sampling: argmax on vocab dimension
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_greedy<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
    let [_b, _v] = logits.dims();
    // argmax(1) keeps dim by default in Burn, resulting in [B, 1]
    let indices = logits.argmax(1);
    debug!("Greedy sample output shape: {:?}", indices.dims());
    indices
}

/// Sample with policy - unified interface
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_with_policy<B: Backend>(
    logits_last: Tensor<B, 2>,
    policy: SamplingPolicy,
    rng: &mut XorShift64,
) -> Tensor<B, 2, Int> {
    use SamplingPolicy::*;

    match policy {
        Greedy => sample_greedy(logits_last),

        Temperature { t } => {
            let logits = apply_temperature(logits_last, t);
            let probs = activation::softmax(logits, 1);
            sample_from_probs(probs, rng)
        }

        TopK { k } => {
            let logits = top_k_filter(logits_last, k);
            let probs = activation::softmax(logits, 1);
            sample_from_probs(probs, rng)
        }

        TopP { p } => {
            let logits = top_p_filter(logits_last, p);
            let probs = activation::softmax(logits, 1);
            sample_from_probs(probs, rng)
        }

        TempTopK { t, k } => {
            let logits = top_k_filter(apply_temperature(logits_last, t), k);
            let probs = activation::softmax(logits, 1);
            sample_from_probs(probs, rng)
        }

        TempTopP { t, p } => {
            let logits = top_p_filter(apply_temperature(logits_last, t), p);
            let probs = activation::softmax(logits, 1);
            sample_from_probs(probs, rng)
        }

        TempTopKTopP { t, k, p } => {
            let logits = top_p_filter(top_k_filter(apply_temperature(logits_last, t), k), p);
            let probs = activation::softmax(logits, 1);
            sample_from_probs(probs, rng)
        }
    }
}

/// Sample with temperature and optional top-k
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_with_temperature_topk<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
    top_k: Option<usize>,
) -> Tensor<B, 2, Int> {
    let mut logits = apply_temperature(logits, temperature);
    if let Some(k) = top_k {
        logits = top_k_filter(logits, k);
    }
    let probs = activation::softmax(logits, 1);
    probs.argmax(1)
}

/// Main entry point for sampling
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_next_token<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
    top_k: Option<usize>,
) -> Tensor<B, 2, Int> {
    if temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_with_temperature_topk(logits, temperature, top_k)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Utility functions
// ═════════════════════════════════════════════════════════════════════════════

/// Extract last timestep logits from [B, T, V] -> [B, V]
pub fn extract_last_logits<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 2> {
    let [b, t, v] = logits.dims();
    logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v])
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling policy enum
// ═════════════════════════════════════════════════════════════════════════════

/// Sampling policy to inject into the engine
#[derive(Clone, Copy, Debug)]
pub enum SamplingPolicy {
    Greedy,
    Temperature { t: f64 },
    TopK { k: usize },
    TopP { p: f64 },
    TempTopK { t: f64, k: usize },
    TempTopP { t: f64, p: f64 },
    TempTopKTopP { t: f64, k: usize, p: f64 },
}
