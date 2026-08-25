# Errata, and a Research Proposal: Is a Learned Structure an Equilibrium or a Fossil?

Supersedes the manifold claims in `claude/distributed-representations-and-formation.md`. Verified against primary sources August 2026.

---

## Part 0 — Errata on what I previously wrote

I wrote: *"what SAEs report as dozens of overlapping features are local parametrizations of one continuous low-dimensional manifold, and attention heads twist manifolds relative to each other."* Checked against the paper (Gurnee, Ameisen, Kauvar, Tarng, Pearce, Olah, Batson, *When Models Manipulate Manifolds*, Transformer Circuits Oct 2025 / arXiv 2601.04480):

| My claim | Actual |
|---|---|
| "dozens of features" | **Ten** — f₁…f₁₀, from a 10M-feature crosscoder. 2 active at a time, 2–3 interpolated. "Dozens" appears once, describing the messy initial pool of vaguely-newline-ish features they waded through during discovery — not the set shown to tile the manifold. ~3–5× inflation. |
| "one manifold" | One **per quantity**. At least four: character count, line width, characters remaining, token length. |
| "low-dimensional" | **Intrinsic dimension 1** (a curve), embedded in an ambient subspace where the **top 6 PCs capture 95% of variance**. "6-dimensional manifold" would be wrong. |
| "local parametrizations" | Their words: features *"tile the manifold in a canonical way, providing approximate local coordinates."* Never "chart" or "atlas." Reconstruction has *"mild kinks… reminiscent of a spline approximation"* — a 10-knot spline, not a smooth atlas. |
| "attention heads twist manifolds" | **Correct — "twist" is their word**, in the abstract and section heading ("Twisting with QK"). But: it is **QK, not OV** (OV does accumulation); and they **do not decompose QK into a rotation** — the rotation lives in an appendix idealization (circulant/Fourier construction), while the empirical claim is offset-alignment shown via probe-transformed cosine similarity. Boundary heads were **never causally ablated**; the interventions target the characters-remaining subspace, not the heads. |
| implied typicality | They hedge explicitly: *"The crispness of the representations and circuits we found may be due to how well the model does the task"* and *"Deep mechanistic case studies benefit from choosing behaviors that the model performs consistently well."* Selected for tractability. |
| implied anti-SAE | **Inverts the paper.** *"Dictionary features provide an unsupervised entry point for discovering mechanisms"*; *"It likely would not have been possible to develop this clarity if it were not for the unsupervised sparse features."* The duality claim is hedged: *"**Sometimes,** discrete features (and their interactions) can be equivalently described using continuous feature manifolds."* The cost of features is parsimony (*"a kind of complexity tax"*), not correctness. |
| "the manifold analogue of dictionary learning… the most direct answer" | **It does not exist.** They *call for* it: *"we would like to see unsupervised approaches to detecting them."* Prior art on feature manifolds is credited to Modell et al. An aspiration, not a technique. |
| anything about formation | **Not in the paper.** No training dynamics, no checkpoints. The only formation-adjacent content is normative (§2.5's capacity-vs-distinguishability optimality argument, plus a toy physical simulation showing ripple/ringing *"arise even when the solution is found dynamically"*). |

**The ringing claim was accurate.** *"We use the term 'ringing' in the sense of signal processing, a transient oscillation in response to a sharp peak, such as in the Gibbs Phenomenon"*; *"Ringing, in the manifold perspective, corresponds to interference in the feature superposition perspective."* Mechanism: the desired similarity structure is trivially realizable in 150 dimensions; forcing it into ~6 truncates Fourier coefficients, producing Gibbs-style overshoot.

**Scope.** One model (Claude 3.5 Haiku), one task, synthetic stimuli (*"we synthetically wrapped to have lines of character length k, for k = 20, 40, …, 140"*). Probe R² = 0.985 after layer 1. Causal evidence is real: patching mean activations into a **rank-6 subspace** changes linebreaking behavior; ablating it matters *"only when the next token is a newline."* Strongest result is §6's visual illusions — inserting `@@` without changing line length hijacks the counting heads.

**Replication status.** One real reproduction: Sinii & Balagansky, *Chasing the Counting Manifold in Open LLMs* (t-tech, Feb 2026), across GPT-2, Pythia, Llama-3.1, Gemma, Qwen3.

| Model | Best layer | Probe R² | PC1-3 var | Newline acc |
|---|---|---|---|---|
| Claude 3.5 Haiku (orig.) | 1–2 | 0.985 | 95% (top 6) | — |
| Pythia-160m | 3 | 0.744 | 95.4% | 36.1% |
| Pythia-410m | 6 | 0.724 | 97.4% | 63.8% |
| GPT-2 | 6 | 0.363 | 78.8% | <20% |
| Gemma-2-9B | 11 | 0.868 | 93.9% | 78% |

Confirms: the curve exists across families, helical PCA trajectories, hill-shaped SAE tuning curves. **Contradicts:** *"In contrast to Anthropic's original result, where peaks appeared in very early layers, the best layer is never in the first two layers for any tested model here."* **Never tested:** ringing, boundary heads, twisting, any causal intervention. (Ignore the `ummagumm-a` HF space — it's an incomplete draft whose abstract terminates mid-sentence at *"contrary to the demonstration of Anthropic,"* with no predicate.)

---

## Part 1 — Why "is it path-dependent?" is the wrong question

The obvious formalization of *"what process, acting locally, would leave this residue?"* is path-dependence: do different trajectories through the same data reach different structures? The literature has largely answered this, and the answer is deflationary.

**Path dependence is real but temporally confined.** Frankle, Dziugaite, Roy & Carbin (ICML 2020) is the only clean isolation of data order as the sole variable — identical initialization, differing only in SGD batch order. At init, interpolating between the two runs drives error to chance. They become linearly mode-connected (barrier <2%) after **1.5% of training** (VGG-16/CIFAR-10), **3%** (ResNet-20), **16–20%** (ImageNet). After that, *"the outcome of optimization is determined to a linearly connected minimum."* Kapoor et al. independently find cross-seed representational alignment crystallizes within the first epoch. Two methods, same window.

**And it washes out with scale.** Curriculum effects on final-checkpoint probes are measurable at 14M–70M (±9–14pp), narrow at 410M, largely vanish at 1B. Sequential vs shuffled pretraining converge to the same general capability (~73–74% OLMES) while differing only on time-indexed knowledge.

**Meanwhile function converges but implementation doesn't.** Schiffman (ICML 2026 MI workshop): independently seeded transformers recover ground-truth Markov eigenvalues to ~1%, CCA ≈ 0.99 — *in nearly orthogonal subspaces* (projector overlap 0.02–0.04, principal angles 75–90°). Universal neurons are 1.23% of GPT2-medium, 1.26% of Pythia-160M, and that's under the *maximally favorable* condition (same data, same order, seed-only variation). SAEs on identical activations differing only in seed share 30–42% of features. Middle-layer attention heads are the least stable across 50 refits (≈0.70 vs ≈1.0 at the edges).

**The live question is different and stronger: endpoint *insufficiency*.** The endpoint can be uniquely determined by the data and still be lossy about its own construction.

- **Transient ridge** (Carroll, Hoogland, Farrugia-Roberts & Murfet, Jan 2025): transformers on in-context linear regression at high task diversity M pass *through* a ridge-regression phase and abandon it. The endpoint is a clean function of M — no path dependence — yet an entire computational phase is fully occupied and fully erased, invisible at the end.
- **Circuits turn over**: circuit membership at formation time vs. at the end shares only **Jaccard ≈ 0.29–0.33** (Pythia-1B, OLMo-1B, OLMoE).
- **But the residue is partly readable.** Krasheninnikov, Turner & Krueger (Sept 2025): after six-stage sequential fine-tuning, stage centroids lie approximately **on a straight line in training order**; a linear probe separates stage 1 from stage 6 on held-out entities at **~90%**; the signal **survives 30 further epochs of shuffled mixed training**, decaying only to ~60–63%. Caveats: activations not weights, fine-tuning not pretraining, ≤8B, toy dataset, mechanism unexplained.

**And the loss-landscape version has never been asked.** LLC and its refinements are used *prospectively* — measure λ̂ over checkpoints, watch stages appear. The retrospective inverse — *given only the endpoint landscape, what can you infer about the sequence that produced it?* — is unposed in the literature. Timaeus already has the SGLD machinery; λ̂ is a purely local endpoint quantity.

**Reframing.** The right question is not "could it have been otherwise?" but **"does the structure record the conditions under which it formed, or only the conditions it currently sits in?"** Equilibrium or fossil.

---

## Part 2 — The proposal

### The question

> **Is the ripple in a language model's character-count manifold an equilibrium property or a fossil?**

The paper's §2.5 argues rippled manifolds *"optimally trade off capacity constraints (roughly, dimensionality) with maintaining the distinguishability of different scalar values (curvature)."* That is an **equilibrium** claim: the geometry is the solution to a constrained optimization given the capacity currently available.

If it is genuinely an equilibrium, the ripple's parameters (wavelength, amplitude, ambient dimension) should track the local capacity situation *throughout training*, including long after the count representation itself has stopped improving — because the surrounding representation space keeps getting more crowded as the model learns everything else.

If instead the ripple locks in during a formation window and stops responding while the neighbourhood keeps changing, the geometry is a **fossil**: a record of the conditions at formation, carried forward.

Three outcomes, each meaning something different, one measurement.

### Substrate

- **Model: Pythia-410m-deduped**, all **154 checkpoints** (step0; 1,2,4,…,512; then every 1,000 to 143,000 ≈ 2.1B tokens apart). Escalate to **Pythia-1.4b** if the signal warrants.
- **Why Pythia and not OLMo 3**: Pythia is the only suite with **exactly recoverable data order** — pre-shuffled memmaps (`EleutherAI/pile-deduped-pythia-preshuffled`) plus `utils/batch_viewer.py`, with `dataset[step*1024:(step+1)*1024]` giving the exact 1024 sequences seen at step N. OLMo publishes no step→batch map. Worse, **Dolma 3 has no books component at all** and Dolma v1.7 has ~10× less Gutenberg than the Pile — you would be hunting a manifold whose training pressure has been removed.
- **Why the Pile has the pressure**: Gutenberg/PG-19 is **2.17%** of the mix and is hard-wrapped at ~70–72 chars (PG-19's only preprocessing is boilerplate removal — no reflow); Enron Emails 0.14% at ~72. Rock-solid hard-wrapped ≈ **2.3%**; with partially column-constrained sources (arXiv 8.96%, GitHub 7.59%, FreeLaw 6.12%) a realistic **3–10%** of tokens. Michaud et al.'s quantization work found *"Cluster 100: predicting newlines in line length limited text"* as a discrete top-400 skill in a **19M-parameter** Pile model. The pressure is real and cheap to satisfy.
- **Probe data**: follow the original protocol — synthetically wrap prose to fixed width k ∈ {20, 40, …, 140}; ridge/multinomial probes for character offset since last newline (150-way).

### Measurements, per checkpoint

1. **Curve**: probe R² for character count, per layer. (Baseline to beat: 0.724 at layer 6.)
2. **Dimension**: PCA variance profile of probe vectors — intrinsic dim, and ambient dim at 95% variance.
3. **Ripple**: off-diagonal band structure in the probe cosine-similarity matrix. Quantify by DFT of the (circulant-ish) similarity matrix — ripple wavelength = dominant off-DC frequency; amplitude = its power. *This is the novel measurement; nobody has computed it on an open model.*
4. **Neighbourhood crowding**: RankMe / effective rank of the residual stream at the same layer — the "how crowded is this space" covariate.
5. **Exposure**: cumulative hard-wrapped tokens seen before step N, via `batch_viewer.py` + a wrap detector (a hard-wrapped doc has `max(line) ≈ p90(line) ≈ 70–80` with low variance; a reflowed doc has `max` in the hundreds).

### Pre-registered predictions

| | Equilibrium | Fossil | Null |
|---|---|---|---|
| Ripple wavelength after R² plateaus | tracks effective rank | flat | unmeasurable |
| Partial correlation (ripple, rank \| step) post-plateau | significantly ≠ 0 | ≈ 0 | n/a |
| Ambient dim | drifts with crowding | set at formation | n/a |

**Go/no-go pilot (one day):** can you measure ringing at all in Pythia-410m's *final* checkpoint? t-tech explicitly did not test it. If the ringing signature is absent or not stably estimable at step 143k, the primary question is dead — and that itself is a result, since it would mean §2.5's account doesn't transfer out of Haiku. Graceful degradation: fall back to "when does the count manifold form, and does its dimensionality expand or contract" using only probe R² + PC variance, both already validated by the reproduction.

### Secondary legs

- **Seed stability (cheap, novel).** PolyPythias gives 9 extra seeds at 410M × 154 checkpoints *with per-seed pre-shuffled data*. Is the formation step seed-stable? Given Frankle's window this is a direct path-dependence test on a specific structure — and ~10× a trivial cost is still trivial.
- **Data attribution.** Correlate formation timing against cumulative hard-wrapped exposure. Only Pythia permits this.
- **The crossed design.** Data-refined LLC on linebreak-heavy data at 12–20 checkpoints around the window. Does the geometric transition coincide with an LLC critical point? Geometry and landscape make *independent* predictions about the same event: agreement is convergent validity for both, disagreement is diagnostic. Note the precedent — in the stagewise LM work, LLC *decreased* during the stage where previous-token heads formed, a signature no behavioural probe would produce.
- **Causal data test.** open-sci-ref-0.01 trains 0.4B/1.3B at matched compute on Pile vs DCLM vs FineWeb-Edu; Pico-decoder-large (570M, Dolma, released fixed order) is a low-Gutenberg control. Does the manifold weaken as hard-wrapped fraction drops?

### Compute

Anchor: Billa's entire *Geometric Anatomy* project — 380 Pythia checkpoints, 6 sizes, 8 tasks, probes + RankMe + LLC + Hessian — ran in **~231 GPU-hours on one RTX 3090 Ti**.

| Tier | Scope | Cost |
|---|---|---|
| 1 | Probes + PCA + ripple, 154 ckpts, 410m | **10–30 GPU-hr** (~$25–75 spot) |
| 2 | + 9 PolyPythia seeds | ~100–300 GPU-hr |
| 3 | + data-refined LLC, ~15 ckpts | ~150–500 GPU-hr at 410m |
| 4 | + 1.4b replication | ~500–2,000 GPU-hr |

Do **not** store activations — accumulate per-layer Gram matrices online (XᵀX at d=1024 is ~4 MB fp32/layer), then every probe variant is a closed-form solve off the cache. Real bottleneck is checkpoint download (40 GB at 410M); stream and delete. Do **not** attempt weight-refined per-head rLLC across checkpoints — one SGLD run per component × 384 heads is why the rLLC paper used a 2-layer toy model. Data-refined (full weights, restricted data) is the ×1 version and the one this needs.

### Known risks

1. **Signal strength.** Haiku R² 0.985 vs Pythia-410m 0.724. The curve is clean (PC1-3 = 97.4%) but the ripple may not survive. Mitigated by the pilot.
2. **Tokenizer confound.** Character counting is mediated by tokenization; GPT-NeoX-20B tokenizer has variable chars/token. Control by conditioning on token-length distribution.
3. **Layer drift.** Best layer may move across checkpoints. Measure per-layer throughout, don't fix a layer early.
4. **LLC estimator fragility.** SGLD hyperparameters have no defaults and must be tuned per checkpoint; the estimator can silently return results identical to the nβ=0 baseline. Run the nβ=0 diagnostic every time. This is why the LLC leg is tier 3, not tier 1.

### Why it's worth doing

The two halves exist and nobody has joined them. Final-model geometry is characterized (Anthropic; t-tech reproduction on the exact models). Checkpoint-based geometry is characterized *in aggregate* (Li et al. NeurIPS 2025: warmup collapse → entropy-seeking expansion → compression-seeking consolidation; Billa: *"geometry changes first, behavior follows"*). **No one has tracked a specific, theoretically-motivated task manifold across pretraining.** The equilibrium/fossil fork turns that gap into a question with a theory attached that can lose.

---

## Sources

Manifolds: [transformer-circuits](https://transformer-circuits.pub/2025/linebreaks/index.html) · [arXiv 2601.04480](https://arxiv.org/abs/2601.04480) · [t-tech reproduction](https://huggingface.co/spaces/t-tech/manifolds) · [Modell et al. prior art via paper's §7]
Checkpoints/data: [Pythia](https://github.com/EleutherAI/pythia) · [arXiv 2304.01373](https://arxiv.org/abs/2304.01373) · [pile-deduped-preshuffled](https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled) · [PolyPythias](https://arxiv.org/abs/2503.09543) · [The Pile](https://ar5iv.labs.arxiv.org/html/2101.00027) · [PG-19](https://github.com/google-deepmind/pg19) · [Pico](https://arxiv.org/abs/2509.16413) · [open-sci-ref](https://arxiv.org/html/2509.09009v1) · [Olmo 3](https://allenai.org/blog/olmo3)
Formation/geometry over training: [Li et al., Representation Geometry](https://arxiv.org/abs/2509.23024) · [Billa, Geometric Anatomy](https://arxiv.org/abs/2602.15997) · [Quantization Model of Neural Scaling](https://arxiv.org/abs/2303.13506) · [When Do Attention Circuits Form?](https://arxiv.org/html/2606.02378)
Path dependence: [Frankle et al., LMC & LTH](https://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf) · [Carroll et al., Transient Structure](https://arxiv.org/html/2501.17745v2) · [Krasheninnikov et al., Training-Order Recency](https://arxiv.org/html/2509.14223v2) · [Schiffman, Invariant Algorithmic Cores](https://arxiv.org/html/2602.22600v2) · [Universal Neurons](https://arxiv.org/abs/2401.12181) · [SAEs Learn Different Features](https://arxiv.org/html/2501.16615v1) · [Influence Dynamics](https://arxiv.org/html/2510.12071v2)
SLT: [Lau et al., LLC](https://arxiv.org/abs/2308.12108) · [Hoogland et al., Stagewise Development](https://arxiv.org/abs/2402.02364) · [Refined LLC](https://arxiv.org/html/2410.02984v1) · [Sampling benchmark](https://arxiv.org/abs/2507.21449) · [Timaeus sampling guide](https://timaeus.co/research/2026-04-21-sampling-guide) · [Skalse critique](https://www.alignmentforum.org/posts/ALJYj4PpkqyseL7kZ/my-criticism-of-singular-learning-theory)
