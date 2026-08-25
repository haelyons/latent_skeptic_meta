<LeftMouse>## First: I overstated the manifold claim, and you were right to push

I checked it against the paper. Corrections, worst first:

**"Dozens of features" — it's ten.** f₁…f₁₀, from a 10M-feature crosscoder, with 2 active at a time. The word "dozens" appears once in the paper, describing the messy initial pool they waded through while hunting: *"We saw dozens of features that were vaguely about newlines and linebroken text."* That's the discovery anecdote, not the result. I inflated it 3–5×.

**"One manifold" — there are at least four**, one per quantity: character count, line width, characters remaining, token length.

**"Low-dimensional" — be precise or it's wrong.** Intrinsic dimension **1** (it's a curve), embedded in a subspace where the top 6 PCs carry 95% of variance. "6-dimensional manifold" would be a misreading.

**"Local parametrizations"** is softer in the original: features *"tile the manifold in a canonical way, providing approximate local coordinates."* They never say chart or atlas. Reconstruction shows *"mild kinks… reminiscent of a spline approximation"* — a 10-knot spline, not a smooth atlas.

**"Attention heads twist manifolds" — this one holds up.** "Twist" is their word, in the abstract and as a section heading. Two tightenings: it's **QK, not OV** (OV does accumulation), and they **never decompose QK into a rotation** — the rotation is an appendix idealization; the measurement is offset-alignment via probe-transformed cosine similarity. Boundary heads were never causally ablated.

**Two framings I gave you were backwards.** The paper is not anti-SAE — *"It likely would not have been possible to develop this clarity if it were not for the unsupervised sparse features."* Features are correct but expensive; geometry is cheaper *where a parametrization exists*. And the duality claim is hedged: *"**Sometimes**, discrete features… can be equivalently described using continuous feature manifolds."* Also, "the manifold analogue of dictionary learning" **does not exist** — they *call for* it. I described an aspiration as a technique and then called it "the most direct answer."

**Scope, in their words:** *"The crispness of the representations and circuits we found may be due to how well the model does the task."* One model, one task, synthetic stimuli. And there's now one real reproduction (Sinii & Balagansky, Feb 2026) across GPT-2/Pythia/Llama/Gemma/Qwen: the curve replicates (Pythia-410m: R² 0.724, PC1-3 = 97.4%), but **the early-layer finding does not** — *"the best layer is never in the first two layers for any tested model here."* Ringing, boundary heads and twisting have never been independently tested at all.

The ringing claim itself was accurate. Their mechanism: the similarity structure you want is trivially realizable in 150 dimensions; forcing it into ~6 truncates Fourier coefficients and you get Gibbs-phenomenon overshoot. And the bridge sentence is theirs: *"Ringing, in the manifold perspective, corresponds to interference in the feature superposition perspective."*

---

## The loss-landscape approach, plainly

The whole thing is one move: **stop asking what the network computes, and ask what it can't afford to change.**

Take the trained weights. Jiggle them randomly at a controlled temperature. Measure how much the loss goes up. If you can jiggle a lot without hurting the loss, many parameter settings work equally well there — the network isn't really using that capacity. If a small jiggle hurts a lot, the solution is finely tuned. That's it. Everything else is bookkeeping.

The **local learning coefficient** λ makes it a number. Ask: how fast does the volume of "good enough" parameter settings shrink as you tighten "good enough"? Formally $V(\epsilon) \propto \epsilon^\lambda$ — tighten your tolerance 10×, and λ is the shrinkage exponent. Low λ means the good set shrinks slowly, i.e. lots of equivalent ways to be near-optimal, i.e. **low complexity**. In a well-behaved model λ collapses to *d/2*, half the parameter count. So λ generalizes "number of parameters actually in use" to models where parameters are redundant.

**Why not just the Hessian?** Because the Hessian assumes the loss is a bowl and it isn't. If loss goes as $w^4$, the Hessian at the minimum is exactly **zero** — it reports "free direction" when the direction actually costs you something, just sub-quadratically. The true answer is λ = 1/4. And there's a counting argument that settles it: λ is rational and routinely not a half-integer, while rank(H)/2 can only ever be a half-integer. No eigenvalue-counting scheme can equal it in general.

**Why this is a theory of development, not just a metric.** The Bayesian free energy is

$$F_n = \underbrace{nL(w^*)}_{\text{fit}} + \underbrace{\lambda \log n}_{\text{complexity}} + \dots$$

That's BIC with λ replacing d/2. The two terms scale differently in n. Early, the $\lambda\log n$ term dominates and *simple-but-worse* solutions win. Later, $nL$ dominates and *complex-but-better* wins. The crossover is discontinuous — a first-order phase transition. **Prediction: models should build simple structures and then destroy them.** That prediction was confirmed: transformers on in-context linear regression pass through a ridge-regression phase and abandon it. Most interpretability frameworks have nothing to say about why a network would build something and then tear it down.

**How much to trust it.** Less than the theory suggests, and Timaeus says so themselves. The chain has five load-bearing gaps: Watanabe's theorems concern the *Bayesian posterior* but we train with SGD; they're *asymptotic in n*; they assume *realizability*; they need $w^*$ to be a *local minimum of population loss* (it isn't, at an arbitrary checkpoint); and SGLD's convergence proofs are **provably incompatible with degenerate landscapes** — which is the case they're used on. On top of that, their own 2026 guide reports the estimator can **silently return numbers identical to the nβ=0 baseline**, hyperparameters are picked by eyeballing loss traces, and the flagship developmental result is a **3M-parameter, 2-layer, attention-only** model.

The fairest summary of what it currently buys: **the LLC tells you *when* to look; conventional mech interp tells you *what's* there.** That's a real service — temporal segmentation is genuinely hard — but it's less than "a new way to see inside models." Two things push back on the purely-deflationary read: LLC *decreased* during the stage where previous-token heads formed (a signature no behavioural probe would give you), and the refined variants separated two induction heads that looked identical, with the separation independently confirmed by ablation.

---

## The two rhetorical moves, audited

**"Which water molecule is responsible for wetness" — I should retract most of this.** The useful half is that the right level of description may be a macroscopic order parameter. The misleading half is that the analogy imports **homogeneity and interchangeability**, and that's precisely false for trained networks. Water molecules are identical. Attention heads *differentiate* — that's the whole rLLC result, heads starting with similar complexity curves and progressively diverging into functional types. So the analogy smuggles in a defeatism ("don't ask about parts") that the evidence doesn't license. Wetness has no privileged molecules; a transformer does have privileged components. Retire it.

**"The manifold analogue of dictionary learning is the most direct answer"** — this was me promoting a call-to-action into a method. There is no such technique. What actually exists is: a well-worked single case, an aspiration in a discussion section, and prior art (Modell et al.) that the paper itself credits.

---

## The residue question: what I found before posing anything

"What process, acting locally, would leave this residue?" is an inverse problem. The obvious formalization is path-dependence — *could it have been otherwise?* — and that turns out to be largely settled and deflationary.

Frankle et al. (2020) is the only clean isolation of data order as sole variable: identical init, differing only in batch order. At init, interpolating between the two runs drives error to chance. But they become linearly mode-connected after **1.5% of training** (VGG-16), 3% (ResNet-20), 16–20% (ImageNet) — *"from [then] on, the outcome of optimization is determined."* Kapoor et al. independently find cross-seed alignment crystallizes in the first epoch. Two methods, same window. And curriculum effects on final internals are measurable at 14M–70M, narrow at 410M, and largely vanish by 1B.

So path-dependence is real, temporally confined, and shrinking with scale. Building a research program on it is building on sand.

**The live question is different, and it's stronger.** The endpoint can be *uniquely determined by the data* and still be *lossy about its own construction*. The transient ridge result is exactly this: the endpoint is a clean function of task diversity M — no path dependence at all — yet an entire computational phase is fully occupied and fully erased. Circuit membership at formation time versus at the end shares only **Jaccard ≈ 0.29–0.33**. And there's one striking positive: after six-stage sequential fine-tuning, stage centroids lie approximately **on a straight line in training order**, a probe separates stage 1 from stage 6 on held-out entities at ~90%, and the signal **survives 30 further epochs of shuffled mixing**.

That last one says the residue is at least *partly* readable. But it's activations, fine-tuning, ≤8B, toy data — and nobody has asked the loss-landscape version. λ̂ is used prospectively (watch stages appear across checkpoints); the retrospective inverse — *given only the endpoint landscape, what can you infer about the sequence that made it?* — is unposed.

So the reframe: not *"could it have been otherwise?"* but **"does the structure record the conditions under which it formed, or only the conditions it now sits in?"** Equilibrium or fossil.

---

## The question

> **Is the ripple in a language model's character-count manifold an equilibrium property or a fossil?**

The paper's §2.5 says rippled manifolds *"optimally trade off capacity constraints (roughly, dimensionality) with maintaining the distinguishability of different scalar values."* That's an **equilibrium** claim — the geometry solves a constrained optimization given currently-available capacity.

If so, the ripple's parameters should keep tracking the local capacity situation *throughout training*, long after the count representation stops improving — because the surrounding space keeps getting more crowded as the model learns everything else. If instead the ripple locks in during a formation window and stops responding while its neighbourhood keeps changing, the geometry is a fossil.

**Inputs.** Pythia-410m-deduped, all 154 checkpoints. Probe data: prose synthetically wrapped at k ∈ {20,…,140}, 150-way character-offset probes, following the original protocol. Training data via `EleutherAI/pile-deduped-pythia-preshuffled` + `batch_viewer.py`, where `dataset[step*1024:(step+1)*1024]` gives the exact 1024 sequences seen at step N.

Pythia specifically, and not OLMo, for two reasons: it's the only suite with exactly recoverable data order, and **Dolma 3 has no books component at all** while Dolma v1.7 has ~10× less Gutenberg than the Pile — you'd be hunting a manifold whose training pressure has been removed. The Pile's pressure is real: Gutenberg is 2.17% of the mix and hard-wrapped at ~70 chars (PG-19's only preprocessing is boilerplate removal, no reflow), and Michaud et al. found *"predicting newlines in line length limited text"* as a discrete top-400 skill cluster in a **19M-parameter** Pile model.

**Measure per checkpoint:** probe R² (curve), PCA variance profile (dimension), DFT of the probe cosine-similarity matrix (ripple wavelength and amplitude — *this is the novel measurement, never computed on an open model*), RankMe at the same layer (crowding), and cumulative hard-wrapped token exposure.

**Pre-registered fork:** post-plateau, partial correlation of ripple wavelength with effective rank, controlling for step. Significantly ≠ 0 → equilibrium. ≈ 0 → fossil. Unmeasurable → §2.5's account doesn't transfer out of Haiku, which is itself a result.

**Go/no-go, one day:** can you see ringing at all in Pythia-410m's *final* checkpoint? t-tech explicitly didn't test it. If not, the primary question dies and you degrade gracefully to "when does the manifold form, and does its dimensionality expand or contract" using only R² and PC variance — both already validated.

**Cost: 10–30 GPU-hours, ~$25–75 spot** for the core. Anchor: Billa's entire *Geometric Anatomy* project (380 Pythia checkpoints, 6 sizes, 8 tasks, probes + RankMe + LLC + Hessian) ran in ~231 GPU-hours on one 3090 Ti. Don't store activations — accumulate per-layer Gram matrices online and every probe variant becomes a closed-form solve off the cache.

Three cheap extensions worth pre-planning: **PolyPythias** gives 9 more seeds at 410M with per-seed data order, so "is the formation step seed-stable?" is a direct path-dependence test on this specific structure for ~10× a trivial cost. **Data attribution** — correlate formation timing with cumulative hard-wrapped exposure — only Pythia permits. And the **crossed design**: data-refined LLC at ~15 checkpoints around the window, where geometry and landscape make independent predictions about the same event. Agreement is convergent validity for both; disagreement is diagnostic. Keep that at tier 3, though, because the LLC estimator needs per-checkpoint hyperparameter tuning and can silently return the nβ=0 baseline.

The gap this fills: final-model geometry is characterized, and checkpoint geometry is characterized *in aggregate* (warmup collapse → expansion → consolidation; *"geometry changes first, behavior follows"*). **Nobody has tracked a specific, theoretically-motivated task manifold across pretraining.** The equilibrium/fossil fork turns that gap into a question with a theory attached that can lose.

Full errata and the detailed design — measurements, tiers, four named failure modes — are in the project doc.
