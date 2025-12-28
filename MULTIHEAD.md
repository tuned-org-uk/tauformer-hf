# i want the mechanism to work on a single-head as the KV-cache benchmarks will be focusing on that but I want to keep open the possibility of going multihead realtively straightforwardly.

I think that a design principle should be kpet that the attention is working in the domain of the manifold. it is up to the pretraining phase in arrowspace to reduce the dimensions without losing resolution. Once the manifold is generated, the TauModeAttention and TauGpt should work at their best as single-head and multi-head with the given manifold

Yes—treating attention as operating *in the manifold domain* is consistent with your goal, and it pairs very naturally with a single-head baseline while keeping a clean path to multihead later.[^1]

## Single-head “native manifold” mode

For KV-cache benchmarks, the cleanest design is: **force the attention head dimension to equal the manifold dimension** so Q/K/V live in the same space as the Laplacian.[^1]
Concretely, when you load a manifold Laplacian of size $M \times M$, run with `n_head=1`, `n_kv_head=1`, and set `n_embd = M` so `head_dim = n_embd / n_head = M`, and then the sparse Laplacian can be applied directly with no projection layer or ambiguity.[^1]

## Keep multihead straightforward

To keep multihead “relatively straightforward”, keep a clear separation between: (1) model mixing width and (2) manifold scoring width.[^1]
That means the module should explicitly track `manifold_dim = laplacian.rows()` and support two valid regimes: **native** (`head_dim == manifold_dim`) and **projected** (`head_dim != manifold_dim`, with a learned map `head_dim -> manifold_dim` used only for tau/lambda computation).[^1]

## Recommended invariants and constructor behavior

A good principle is: “the Laplacian defines the scoring space, always.”[^1]
So in `new_with_laplacian`, always set `self.manifold_dim = laplacian.rows()` and validate square-ness, and then: if `head_dim == manifold_dim`, use the direct sparse lambda path; otherwise require (and initialize) manifold projection layers for Q/K (and optionally V) so lambdas are computed on `[B,H,T,manifold_dim]` regardless of how many heads you use.[^1]

## Cache implications (single-head first, multihead later)

In single-head native manifold mode, caching stays minimal and stable: cache `V` and `lambda_k` in manifold space, and the cache size grows only with sequence length, which is ideal for your benchmark focus.[^1]
When you later enable multihead, you can keep the same cache structure “per KV head” (especially if you keep MQA/GQA), because `lambda_k` is still computed in the manifold domain and only needs expansion across query heads when `n_head != n_kv_head`.[^1]

If the manifold is always available at model construction time (which it sounds like), the most robust next step is to implement both regimes but **default tests/benchmarks** to `n_head=1, n_embd=manifold_dim` so the mechanism is natively manifold-first with no projection moving parts.
