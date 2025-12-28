use burn::tensor::{Int, Tensor, backend::Backend};
use log::{debug, info};

// Concrete models (renamed for clarity at call sites).
use crate::causalattention::GptModel as NanoModel;
use crate::taugpt::{TauGptModel as TauModel, TauKVCache};

/// Cache trait so Engine can drive either KV cache implementation. [file:47][file:46]
pub trait GptCache<B: Backend>: Sized {
    fn new(n_layer: usize) -> Self;
    fn clear(&mut self);
    fn position(&self) -> usize;
    fn advance(&mut self);
}

/// Model trait (generic “GPT-like” API: prefill + decode). [file:47][file:46]
pub trait GptModel<B: Backend>: Sized {
    type Cache: GptCache<B>;

    fn num_layers(&self) -> usize;

    fn forward(&self, input_ids: Tensor<B, 2, Int>, use_softcap: bool) -> Tensor<B, 3>;

    fn forward_decode(
        &self,
        last_id: Tensor<B, 2, Int>, // [B,1]
        cache: &mut Self::Cache,
        use_softcap: bool,
    ) -> Tensor<B, 3>;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT KVCache (existing implementation), now implementing GptCache
// ─────────────────────────────────────────────────────────────────────────────

/// Simple per-layer KV cache.
/// Shapes:
///   K: [B, H_kv, T_total, D]
///   V: [B, H_kv, T_total, D]
pub struct KVCache<B: Backend> {
    pub(crate) store: Vec<Option<(Tensor<B, 4>, Tensor<B, 4>)>>,
    t_pos: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn get(&self, layer_idx: usize) -> Option<&(Tensor<B, 4>, Tensor<B, 4>)> {
        self.store[layer_idx].as_ref()
    }

    pub fn append_step(
        &mut self,
        layer_idx: usize,
        k_step: Tensor<B, 4>, // [B, H_kv, 1, D]
        v_step: Tensor<B, 4>, // [B, H_kv, 1, D]
    ) {
        let [b, h, t_new, d] = k_step.dims();
        debug!(
            "KVCache: append_step layer={} step_dims(K)=[B={},H={},T={},D={}] t_pos={}",
            layer_idx, b, h, t_new, d, self.t_pos
        );

        match &mut self.store[layer_idx] {
            Some((k_all, v_all)) => {
                let t_prev = k_all.dims()[2];
                debug!(
                    "KVCache: concatenating existing K/V at layer {} (prev T={} -> new T={})",
                    layer_idx,
                    t_prev,
                    t_prev + t_new
                );
                let new_k = Tensor::cat(vec![k_all.clone(), k_step], 2);
                let new_v = Tensor::cat(vec![v_all.clone(), v_step], 2);
                *k_all = new_k;
                *v_all = new_v;
            }
            slot @ None => {
                debug!(
                    "KVCache: initializing layer {} with first K/V chunk (T={})",
                    layer_idx, t_new
                );
                *slot = Some((k_step, v_step));
            }
        }
    }
}

impl<B: Backend> GptCache<B> for KVCache<B> {
    fn new(n_layer: usize) -> Self {
        info!("KVCache: initializing with {} layers", n_layer);
        Self {
            store: vec![None; n_layer],
            t_pos: 0,
        }
    }

    fn clear(&mut self) {
        info!("KVCache: clearing all layers and resetting position");
        for (i, slot) in self.store.iter_mut().enumerate() {
            if slot.is_some() {
                debug!("KVCache: clearing layer {}", i);
            }
            *slot = None;
        }
        self.t_pos = 0;
    }

    fn position(&self) -> usize {
        self.t_pos
    }

    fn advance(&mut self) {
        self.t_pos += 1;
        debug!("KVCache: advanced position to t_pos={}", self.t_pos);
    }
}

// TauKVCache adapter (so Engine can also drive TauModel). [file:46]
impl<B: Backend> GptCache<B> for TauKVCache<B> {
    fn new(n_layer: usize) -> Self {
        TauKVCache::new(n_layer)
    }

    fn clear(&mut self) {
        self.reset();
    }

    fn position(&self) -> usize {
        self.position
    }

    fn advance(&mut self) {
        self.position += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Implement GptModel trait for NanoModel and TauModel
// ─────────────────────────────────────────────────────────────────────────────

impl<B: Backend> GptModel<B> for NanoModel<B> {
    type Cache = KVCache<B>;

    fn num_layers(&self) -> usize {
        self.num_layers()
    }

    fn forward(&self, input_ids: Tensor<B, 2, Int>, use_softcap: bool) -> Tensor<B, 3> {
        self.forward(input_ids, use_softcap)
    }

    fn forward_decode(
        &self,
        last_id: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
        use_softcap: bool,
    ) -> Tensor<B, 3> {
        self.forward_decode(last_id, cache, use_softcap)
    }
}

impl<B: Backend> GptModel<B> for TauModel<B> {
    type Cache = TauKVCache<B>;

    fn num_layers(&self) -> usize {
        self.num_layers()
    }

    fn forward(&self, input_ids: Tensor<B, 2, Int>, use_softcap: bool) -> Tensor<B, 3> {
        self.forward(input_ids, use_softcap)
    }

    fn forward_decode(
        &self,
        last_id: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
        use_softcap: bool,
    ) -> Tensor<B, 3> {
        self.forward_decode(last_id, cache, use_softcap)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic Engine
// ─────────────────────────────────────────────────────────────────────────────

pub struct Engine<B: Backend, M: GptModel<B>> {
    pub model: M,
    _device: B::Device,
}

impl<B: Backend, M: GptModel<B>> Engine<B, M> {
    pub fn new(model: M, device: B::Device) -> Self {
        info!("Engine: new with {} layers", model.num_layers());
        Self {
            model,
            _device: device,
        }
    }

    pub fn prefill(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = input_ids.dims();
        info!("Engine: prefill forward [B={},T={}]", b, t);
        let logits = self.model.forward(input_ids, true);
        debug!("Engine: prefill logits shape {:?}", logits.dims());
        logits
    }

    pub fn decode_next(
        &self,
        last_id: Tensor<B, 2, Int>, // [B,1]
        cache: &mut M::Cache,
    ) -> Tensor<B, 3> {
        let [b, t] = last_id.dims();
        debug_assert_eq!(t, 1, "decode_next expects [B,1], got T={}", t);
        info!(
            "Engine: decode_next at t_pos={} [B={},T=1]",
            cache.position(),
            b
        );
        let logits = self.model.forward_decode(last_id, cache, true);
        debug!("Engine: decode_next logits_step shape {:?}", logits.dims());
        logits
    }

    pub fn stream<'a>(
        &'a self,
        ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
    ) -> Streamer<'a, B, M> {
        let [b, t] = ids.dims();
        info!(
            "Engine: streaming start [B={},T0={}] max_new_tokens={}",
            b, t, max_new_tokens
        );
        let cache = M::Cache::new(self.model.num_layers());
        Streamer {
            engine: self,
            ids: Some(ids),
            cache,
            steps_left: max_new_tokens,
            finished: false,
        }
    }

    pub fn decode_next_with_policy(
        &self,
        last_id: Tensor<B, 2, Int>,
        cache: &mut M::Cache,
        policy: crate::sampling::SamplingPolicy,
        rng: &mut crate::sampling::XorShift64,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 3>) {
        let logits_step = self.decode_next(last_id, cache); // [B,1,V]
        let [b, _, v] = logits_step.dims();

        // sample_with_policy expects [B,V] and returns [B,1]
        let next =
            crate::sampling::sample_with_policy(logits_step.clone().reshape([b, v]), policy, rng);

        (next, logits_step)
    }

    /// Multi-sample generation (batch = num_samples).
    ///
    /// Notes:
    /// - This expands the prompt to shape [num_samples, T0]. [file:47]
    /// - It primes the decode cache by feeding the prompt token-by-token (except the last),
    ///   advancing cache position after each step (same pattern as Streamer). [file:47]
    /// - Sampling uses `sample_with_policy` (currently deterministic for non-greedy policies
    ///   because it ends up using argmax after softmax). [file:19]
    pub fn generate_multi(
        &self,
        prompt: &[i64],
        num_samples: usize,
        max_new_tokens: usize,
        policy: crate::sampling::SamplingPolicy,
        seed: u64,
    ) -> Tensor<B, 2, Int> {
        assert!(num_samples > 0, "num_samples must be > 0");
        assert!(!prompt.is_empty(), "prompt must be non-empty");

        let device = &self._device;
        let t0 = prompt.len();

        // Build ids [num_samples, T0] by repeating the prompt.
        let mut rep: Vec<i64> = Vec::with_capacity(num_samples * t0);
        for _ in 0..num_samples {
            rep.extend_from_slice(prompt);
        }
        let mut ids =
            Tensor::<B, 1, Int>::from_ints(rep.as_slice(), device).reshape([num_samples, t0]);

        // Init + clear cache.
        let mut cache = M::Cache::new(self.model.num_layers());
        cache.clear();

        // Prime cache with the prompt (decode all tokens except the last).
        if t0 > 1 {
            for pos in 0..(t0 - 1) {
                let last = ids.clone().slice([0..num_samples, pos..pos + 1]); // [B,1]
                let _ = self.decode_next(last, &mut cache);
                cache.advance();
            }
        }

        // RNG for stochastic sampling (lightweight, deterministic by seed).
        let mut rng = crate::sampling::XorShift64::new(seed);

        // Generate.
        for _ in 0..max_new_tokens {
            let [b, t] = ids.dims();
            let last = ids.clone().slice([0..b, (t - 1)..t]); // [B,1]

            let (next, _logits_step) =
                self.decode_next_with_policy(last, &mut cache, policy, &mut rng);

            ids = Tensor::cat(vec![ids, next], 1);
            cache.advance();
        }

        ids
    }
}

// Streaming iterator that yields one token id [B,1] each step.
pub struct Streamer<'a, B: Backend, M: GptModel<B>> {
    engine: &'a Engine<B, M>,
    ids: Option<Tensor<B, 2, Int>>,
    cache: M::Cache,
    steps_left: usize,
    finished: bool,
}

impl<'a, B: Backend, M: GptModel<B>> Iterator for Streamer<'a, B, M> {
    type Item = Tensor<B, 2, Int>; // [B,1]

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.steps_left == 0 {
            info!(
                "Streamer: finished (steps_left={}, finished={})",
                self.steps_left, self.finished
            );
            self.finished = true;
            return None;
        }
        let ids = self.ids.as_ref().unwrap();

        let [b, t] = ids.dims();
        debug!("Streamer: step begin [B={},T_current={}]", b, t);
        let last_id = ids.clone().slice([0..b, (t - 1)..t]); // [B,1]

        let logits_step = self.engine.decode_next(last_id, &mut self.cache); // [B,1,V]
        let [_, _, v] = logits_step.dims();

        let next = logits_step.reshape([b, v]).argmax(1).reshape([b, 1]);

        self.ids = Some(Tensor::cat(vec![ids.clone(), next.clone()], 1));
        self.cache.advance();
        self.steps_left -= 1;

        info!(
            "Streamer: emitted next token, steps_left={}, t_pos={}",
            self.steps_left,
            self.cache.position()
        );
        Some(next)
    }
}

/// Convenience aliases.
pub type NanoEngine<B> = Engine<B, NanoModel<B>>;
pub type TauEngine<B> = Engine<B, TauModel<B>>;
