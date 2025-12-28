//
// Tests for TauGptModel (taumode attention GPT) using the sparse constructor only.
//
// Design principle (native manifold-first):
// - In sparse mode, the manifold Laplacian is available at model construction time.
// - Attention should operate in the manifold domain.
// - Therefore, for the default benchmark regime we use:
//     n_head = 1, n_kv_head = 1, n_embd = manifold_dim
//   so that head_dim == manifold_dim and no projection is needed.
//
// Also includes a lightweight multi-head check that remains "manifold-native" by ensuring
// head_dim == manifold_dim via n_embd = n_head * manifold_dim.

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use std::path::Path;

use crate::backend::AutoBackend;
use crate::config::NanoChatConfig;
use crate::pretraining::parquet::{DomainManifold, load_domain_manifold};
use crate::taugpt::{TauGptModel, TauKVCache};

type B = AutoBackend;

// Path to your manifold parquet used by the sparse constructor.
use rstest::fixture;
use rstest::rstest;

const MANIFOLD_PATH: &str = "./domain_manifold/manifold.parquet";

#[fixture]
#[once]
fn load_manifold_test() -> (DomainManifold, usize) {
    let domain_manifold = load_domain_manifold(MANIFOLD_PATH).unwrap();
    let manifold_dim = domain_manifold.nfeatures;

    (domain_manifold, manifold_dim)
}

fn cfg_sparse_native_single_head() -> NanoChatConfig {
    let (_, manifold_dim) = load_manifold_test();
    NanoChatConfig {
        sequence_len: 16,
        vocab_size: 64,
        n_layer: 2,

        // Native manifold-first single-head (KV-cache benchmark regime)
        n_head: 1,
        n_kv_head: 1,
        n_embd: manifold_dim, // head_dim = manifold_dim

        block_size: 16,
        dropout: 0.0,
    }
}

fn cfg_sparse_native_multihead(n_head: usize, n_kv_head: usize) -> NanoChatConfig {
    assert!(n_head >= 1);
    assert!(n_kv_head >= 1);
    assert!(n_kv_head <= n_head);
    assert!(n_head % n_kv_head == 0);

    let (_, manifold_dim) = load_manifold_test();

    NanoChatConfig {
        sequence_len: 16,
        vocab_size: 64,
        n_layer: 2,

        // Still manifold-native per head:
        // head_dim = n_embd / n_head = manifold_dim
        n_head,
        n_kv_head,
        n_embd: n_head * manifold_dim,

        block_size: 16,
        dropout: 0.0,
    }
}

fn build_model(cfg: &NanoChatConfig, device: &<B as Backend>::Device) -> TauGptModel<B> {
    let tau_mode = crate::pretraining::parquet::TauMode::Median;

    let gpt = TauGptModel::<B>::new_with_sparse_laplacian(
        cfg,
        &Path::new(MANIFOLD_PATH),
        device,
        tau_mode,
    );

    // In the "manifold-native" regime we always want head_dim == manifold_dim.
    // For multihead, this is enforced by cfg_sparse_native_multihead via n_embd = n_head * manifold_dim.
    let head_dim = cfg.n_embd / cfg.n_head;
    assert_eq!(cfg.n_embd % cfg.n_head, 0);
    assert_eq!(
        head_dim,
        gpt.laplacian_dims(),
        "Tests assume manifold-native head_dim == MANIFOLD_DIM"
    );

    gpt
}

fn assert_logits_finite(logits: &Tensor<B, 3>) {
    let v: Vec<f32> = logits.clone().to_data().to_vec().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "Found NaN/Inf in logits");
}

fn max_abs_diff(a: Tensor<B, 3>, b: Tensor<B, 3>) -> f32 {
    let diff = (a - b).abs();
    let v: Vec<f32> = diff.to_data().to_vec().unwrap();
    v.into_iter().fold(0.0f32, |m, x| m.max(x))
}

fn run_forward_decode_equivalence(cfg: NanoChatConfig) {
    let device = <B as Backend>::Device::default();
    let model = build_model(&cfg, &device);

    let bsz = 2usize;
    let t = 8usize;

    // Deterministic token pattern (no RNG dependency).
    let ids: Vec<i64> = (0..(bsz * t))
        .map(|i| (i as i64) % (cfg.vocab_size as i64))
        .collect();

    let idx = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([bsz, t]);

    // Prefill forward logits: [B,T,V]
    let logits_fwd = model.forward(idx.clone(), false);
    let v = logits_fwd.dims()[2];

    assert_eq!(logits_fwd.dims(), [bsz, t, cfg.vocab_size]);
    assert_logits_finite(&logits_fwd);

    // Decode the same sequence token-by-token and compare logits at each position.
    let mut cache = TauKVCache::<B>::new(model.num_layers());
    cache.reset();

    for pos in 0..t {
        let last = idx.clone().slice([0..bsz, pos..pos + 1]); // [B,1]
        let logits_step = model.forward_decode(last, &mut cache, false); // [B,1,V]

        assert_eq!(logits_step.dims(), [bsz, 1, v]);
        assert_logits_finite(&logits_step);

        let logits_fwd_pos = logits_fwd.clone().slice([0..bsz, pos..pos + 1, 0..v]); // [B,1,V]
        let mad = max_abs_diff(logits_step, logits_fwd_pos);

        assert!(
            mad < 1e-6,
            "forward/decode logits mismatch at pos={}, max_abs_diff={}",
            pos,
            mad
        );
    }
}

#[rstest]
fn test_taugpt_construction_sparse_native_single_head() {
    let cfg = cfg_sparse_native_single_head();
    let device = <B as Backend>::Device::default();
    let _model = build_model(&cfg, &device);
}

#[rstest]
fn test_taugpt_forward_shape_and_finite_sparse_native_single_head() {
    let cfg = cfg_sparse_native_single_head();
    let device = <B as Backend>::Device::default();
    let model = build_model(&cfg, &device);

    let bsz = 3usize;
    let t = 6usize;

    let ids: Vec<i64> = (0..(bsz * t))
        .map(|i| (i as i64) % (cfg.vocab_size as i64))
        .collect();
    let idx = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([bsz, t]);

    let logits = model.forward(idx, true);
    assert_eq!(logits.dims(), [bsz, t, cfg.vocab_size]);
    assert_logits_finite(&logits);
}

#[rstest]
fn test_taugpt_generation_determinism_sparse_native_single_head() {
    let cfg = cfg_sparse_native_single_head();
    let device = <B as Backend>::Device::default();
    let model = build_model(&cfg, &device);

    let prompt: Vec<i64> = vec![1, 2, 3, 4];
    let idx = Tensor::<B, 1, Int>::from_ints(prompt.as_slice(), &device).reshape([1, prompt.len()]);

    let out1 = model.generate(idx.clone(), 8);
    let out2 = model.generate(idx, 8);

    let v1: Vec<i64> = out1.to_data().to_vec().unwrap();
    let v2: Vec<i64> = out2.to_data().to_vec().unwrap();
    assert_eq!(v1, v2, "Greedy generation should be deterministic");
}

#[rstest]
fn test_taugpt_forward_decode_matches_prefill_sparse_native_single_head() {
    run_forward_decode_equivalence(cfg_sparse_native_single_head());
}

#[rstest]
fn test_taugpt_forward_decode_matches_prefill_sparse_native_multihead_no_mqa() {
    // Multihead, still manifold-native per head (head_dim == manifold_dim).
    // Keep heads small to avoid slowing unit tests too much.
    run_forward_decode_equivalence(cfg_sparse_native_multihead(2, 2));
}

#[rstest]
fn test_taugpt_forward_decode_matches_prefill_sparse_native_multihead_mqa() {
    // Multihead with MQA (n_head != n_kv_head), still manifold-native per head.
    run_forward_decode_equivalence(cfg_sparse_native_multihead(2, 1));
}
