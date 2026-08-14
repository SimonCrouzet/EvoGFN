# What this does not show

This page exists because the rest of the documentation is easier to trust if the limits are
stated in one place rather than buried where each of them happens to arise.

Nothing here is hypothetical. Every entry is a negative result from this project's own benchmark
suite, a control that was not run, a scope the evidence does not reach, or a defect in the
machinery that produces the numbers.

**Recomputed from `results/` on 2026-08-14.** The previous version of this page was written when
the store held 7,040 records and no headline GFlowNet campaign existed. Both facts have changed,
and several entries that follow are corrections to things this page itself asserted.

---

## 1. Every constraint claim in this project rests on one generator family

This is the largest limit and the one least fixable by more compute.

`gb1-anchor` and `trpb-anchor` are empirical four-site landscapes with **no feasibility constraint
at all**. Every task that constrains construction — `feasibility`, `protocol-alde`,
`protocol-evolvepro`, the six `replicate-*` instances, `large-space` — is an **Ehrlich** function.
So "masking changes what a benchmark can reach" is demonstrated on Ehrlich and nowhere else.

The audit against `pytorch-holo` narrows this but does not remove it: it shows the finding is a
property of the *reference implementation* rather than of our port, which is a different and
stronger claim than generality across constraint families. A reader wanting "does this happen for
RNA secondary structure, or for a learned foldability classifier" will not find it here.

**Effective independence is smaller than the seed counts suggest.** `protocol-alde` and
`protocol-evolvepro` are the same Ehrlich instance under two protocols; the six `replicate-*` tasks
are the other instances. The headline comparison spans **four instances × two protocols**, not
eight independent landscapes. Seed counts of 100 buy precision within an instance and do not buy
instance-level generality.

---

## 2. `large-space` has no GFlowNet campaign

Zero stored records for any GFlowNet arm on `large-space` (L=256), against 30 seeds each for
`cmaes`, `genetic` and the other classical arms. It is the only task in the suite where the
sequence length is an order of magnitude above the rest, and therefore the only place the question
"does the margin grow with L" could be asked. It is unanswered.

The holo audit *does* cover L=256 as an instance property, but the campaign comparison does not.

---

## 3. Two arms are ours, not the literature's, and must never be quoted as published methods

**`cmaes+dp` is not a baseline.** It is CMA-ES handed an *exact* constrained decoder written here;
`benchmark/methods.py` and `docs/benchmark.md` both say it must not enter a table reporting
published methods. Its legitimate use is as the **exact-projection instrument**: the thing the
learned sampler is measured against to ask whether an exact solve suffices.

**A correction this page previously got wrong.** An earlier reading held that a non-learned method
beat the GFlowNet on diversity on 3 of 3 tasks. That comparison was against `cmaes+dp`. Against the
**published** `cmaes` arm, 100 paired seeds:

| task | `cmaes` − gfn diversity | verdict |
|---|---|---|
| `protocol-alde` | −10.725, 0/0/100 | GFlowNet wins |
| `feasibility` | 0/1/99 | GFlowNet wins |
| `protocol-evolvepro` | +0.171, p=0.19 | tie |
| `gb1-anchor` | +0.140, 99/0/1 | GFlowNet loses |

So against published methods the GFlowNet wins two, ties one, and loses one — and the single loss
is `gb1-anchor`, where there is no feasibility constraint, so the reachable set *is* the Hamming
ball *is* the whole 160,000-variant library and support correction is provably a no-op there.

**`mlde+earlyfit` is ours.** It lowers MLDE's training size so the supervised handover happens
within this suite's budget. The value comes from this suite's own measured feasible share and
appears nowhere in Wittmann et al. It must never be quoted as MLDE.

---

## 4. MLDE never reaches its supervised stage on a constrained task

`fitted` is recorded per seed, so this is visible rather than inferred:

| task | `mlde` fitted | `mlde+earlyfit` fitted |
|---|---|---|
| `protocol-alde` | **0 / 100** | 84 / 100 |
| `feasibility` | **0 / 100** | 100 / 100 |
| `gb1-anchor` | 3 / 100 | 100 / 100 |

Both published MLDE arms gate the handover on *usable* measurements and discard infeasible assays,
so on a landscape where most wells return nothing the supervised phase is unreachable at any budget
a laboratory would run. Those rows are a finding about the regime, not a tuned comparison — but
they are not a measurement of MLDE-as-published performing badly, because MLDE-as-published never
ran.

---

## 5. The oracle budget is the only budget, and it is identical everywhere

Every arm on every task spends **384 oracle calls**. Nothing in this suite measures a method that
buys a better result with more assays, because no such comparison exists here.

What differs between arms is *compute against a free surrogate*, and it differs enormously:

| arm (`protocol-alde`) | proxy calls | CPU |
|---|---|---|
| `genetic` | 0 | 0.05 s |
| `cmaes` | 0 | 0.1 s |
| `cmaes+dp` | 0 | 3.1 s |
| `alde`, `mlde` | 0 | 13.2 s, 2.2 s |
| `adalead` | 5,254 | 14.0 s |
| GFlowNet | **38,400** | **332.8 s** |

**`proxy_calls` counts training evaluations only** — the per-round screening pool is not counted in
it. Any statement of the form "N× fewer proxy calls" is therefore about *training* compute, and is
not a budget saving in the sense a laboratory means. This page previously implied otherwise.

The 72–202× CPU gap is real and is a genuine cost of the method. It is not offset by anything
measured here.

---

## 6. The reward lattice is coarser than most effects

On `protocol-alde` the `best` column takes **seven distinct values across all arms**, with a
minimum step of 0.0625 — larger than every within-cluster effect in the suite. Any single arm sees
fewer: `genetic` reaches three. Means and standard errors on that column are therefore misleading,
and this page reports **sign tests and W/T/L** instead wherever the comparison is close.

A corollary that has bitten before: a ratio between two effects on this lattice is a ratio of a
signal to a detection limit, not a ratio of effects. No such ratio is quoted here.

---

## 7. What the untrained control does and does not settle

The decomposition rungs run the shipped GFlowNet with **no training at all** (`steps=0`, random
initialisation), so they draw from the same masked construction graph and differ in learning alone.
Paired on `best`, 100 seeds, W/T/L = GFlowNet better / tie / untrained better:

| task | matched proposals | 4× proposals | 16× |
|---|---|---|---|
| `protocol-alde` | 28/62/10, p=0.005 | 41/45/14, p<0.001 | *n=8, incomplete* |
| `protocol-evolvepro` | 35/52/13, p=0.002 | 35/49/16, p=0.011 | *not run* |
| `feasibility` | 0/100/0 | 0/100/0 | 0/100/0 |
| `gb1-anchor` | 63/20/17, p<0.001 | 52/3/45, p=0.54 | 47/2/51, p=0.76 |
| `trpb-anchor` | 67/17/16, p<0.001 | 52/3/45, p=0.54 | 45/3/52, p=0.54 |

**What this establishes.** Where the constraint binds, training is load-bearing and more proposals
do not substitute for it. Where it does not (GB1, TrpB), the untrained policy catches up at 4×
proposals, so training buys sample efficiency and nothing else.

**What it does not establish.** On `feasibility` every arm ties exactly at 0/100/0 — the task is
solved by masking alone and separates nothing. The 16× column is incomplete on both protocol tasks.

**A control that was specified and then deliberately changed.** An earlier plan named a
`masked-screen@matched-proxy` arm that would equalise *surrogate spend*. It was not built, and that
is a decision rather than an omission: the surrogate is not the campaign's budget (§5), so pegging a
control to the trained arm's surrogate consumption would handicap it for no principled reason. The
proposal sweep above answers the question that survives — how far masking plus generous screening
gets on its own. A reader who wants the proxy-matched comparison should know it does not exist.

---

## 8. A rejection-cost column was wrong, and was re-run rather than blessed

The stored `genetic-feasible` records reported `draws_attempted = 0` on every seed of every
constrained Ehrlich task. That was not a rejection rate of zero; it was a column added after those
records were written, waved forward by a bless that was correct about *results* and silently wrong
about a field the old records could not carry.

Re-run, 100 seeds, corrected:

| task | attempted | rejected | rate |
|---|---|---|---|
| `protocol-alde` | 3,340 | 1,704 | **51.0%** |
| across all 8 constrained instances | 2,968–4,052 | 1,394–2,153 | **47–53%** |

A rejection-sampling GA discards roughly half of what it draws on every constrained instance, while
a masked policy discards none. Note the direction: the stored data *understated the comparator's
cost*, so this error worked against this project's own argument.

Read the `draws_unmutated` column beside it (156–377). Those draws carry no substitution at all, are
trivially feasible, and hold the apparent rejection rate *down*.

**Scope of the exposure.** A survey of all 616 cells for the same signature — an arm recording draws
on some tasks and zero on others — returns `genetic-feasible` and nothing else.

---

## 9. Records are currently stale, by construction

The store holds **41,274 campaign records**; **1,808** are current against today's source
fingerprint. That ratio is not decay: a refactor of the feasibility seam landed on 2026-08-13 and
touched modules inside the fingerprinted closure, which marks every record stale at a stroke.

The remedy is a *bless* — restamping records for a module whose change provably cannot have moved
them — and a bless is only honest if somebody checked. `tests/benchmark/test_record_equivalence.py`
re-runs committed cells through the production path and asserts the fresh record agrees on every
field that is not a clock or a fingerprint. Until that bless is performed and recorded, **treat
every number on this page as measured under the source that produced it and not necessarily under
`HEAD`**.

That test found one real regression (feasibility silently skipped from round two after re-anchoring)
and one stale-data error (§8). It also missed a class of failure until a cell combining re-anchoring
*and* feasibility filtering was added — a reminder that a roster is only as good as the axes it
crosses.

---

## 10. Determinism is enforced, and it is enforced by checking

Threading is pinned before any tensor work, and `is_deterministic` interrogates the **live** thread
pools rather than trusting that the pinning call ran — because the two disagreed in the case that
actually occurred, where the limit reached torch and never reached numpy's BLAS. Every stored record
carries the flag.

What this does not cover: a library imported after pinning brings its own pool. The environment
variables catch the common case; the flag on the record is what a reader should check.

---

## 11. The multi-objective result is narrower than its section title

At a fixed 384-assay budget, one preference-conditioned policy matches eight separately-trained
scalarised policies. Every qualification below belongs beside that sentence:

* **The assay multiplier is exactly 1.** The plate is split `batch_size // preferences`; no arm
  spends more oracle calls than another. The saving is 8× *proxy calls* and **3.4×** CPU.
* **Only the eight-preference comparison is grid-matched.** `preference_vectors` returns
  `linspace(0,1,K)` with a special case at K=1 returning the neutral (0.5, 0.5), and the conditioned
  arm is always queried at the K=8 grid. So conditioned-versus-one-preference compares a method
  covering eight front points against one covering one — a hypervolume artefact. That is the only
  significant row, and it is the confounded one. Grid-matched, it is a tie (42/22/36, p=0.57).
* **K was never swept past 8**, and the protocol structurally cannot pass ~16, since `96 // K`
  assays per preference degenerates.
* **Two objectives**, so the simplex is an interval — the geometry where conditioning is easiest.
* **Weighted-sum scalarisation cannot reach a concave front at any preference**, so "one model
  covers the front" is bounded by the scalarisation rather than by the conditioning.
* **NSGA-II runs at 4 rounds of 96 rather than 250 generations**, a regime it was not designed for.
  Comparisons against it carry no weight and are not quoted.

---

## 12. Diversity is measured against the wrong support, and the fix is not free

`diversity` is the mean pairwise Hamming distance over everything measured, **normalised by
nothing**. Under re-anchoring the campaign-level number sums within-round spread with how far the
anchor travelled, which is not a quantity with a well-defined support.

Decomposing from stored records, on `protocol-evolvepro` the `cmaes+dp` − gfn gap is +1.488 overall
but only **+0.097 within-round** — the remainder is the anchor wandering 6.3 Hamming units further
over eight rounds.

Two things a support-corrected metric cannot do, stated so nobody expects them: normalising by a
per-task constant **cannot change a sign**, and on `gb1-anchor` — the one task where the GFlowNet
cleanly loses to a published method — there is no constraint, so support correction is a no-op by
construction. A metric that could move a sign (coverage, or a Hill number over the mutated-position
partition) needs the evaluated batch, which older records do not store.

---

## 13. Theory that is derived here rather than cited

The formal claims underpinning the soundness condition are of mixed provenance, and the difference
matters:

* **Verified by exhaustive check.** The NP-completeness of monotone reachability under an
  integer-weighted budget band, by reduction from PARTITION — 1,977 instances, both directions,
  zero mismatches.
* **Verified against the reference.** The completion-oracle mask recovering `F ∩ B_B` exactly, on
  32 of 32 instance–budget cells.
* **Derived here, not cited.** The temporal-CSP reformulation, the majority-polymorphism
  generalisation of Gopalan et al.'s Lemma 4.3, and the claimed NP-completeness of 3-CNF monotone
  reachability. The last rests on one operation definition reconstructed from a garbled source and
  **must not carry a load-bearing claim** until checked directly.

Three errors caught before they reached a draft, recorded because they are easy to repeat:
PSPACE-completeness **cannot** hold for this reachable set (the flip order is a polynomial
certificate, so it is in NP); one-hot encoding a v>2 alphabet into Boolean CNF **freezes the entire
solution graph**, since flipping any single Boolean breaks exactly-one; and "binary CSP" is not the
analogue of 2-CNF — *majority-closed* is.

---

## 14. A latent counting error, scoped

`log_n_trajectories` returns `log k!`, which **over-counts under intermediate masking**:
`backward_mask` restricts to feasible parents, so the true parent count is at most `k`.

Checked rather than assumed: the method is called from two tests and from nowhere in `src/`, and the
exact-L1 measurement does not use it — that quantity is an empirical histogram of drawn samples over
the reachable set against a reward-proportional target. **No published number here depends on it.**
It is exact under the completion-oracle mask and under terminal-only feasibility, and it would be
wrong under intermediate masking, so any future exact-`P_B` derivation must not build on it.

---

## 15. Novelty, stated against what already exists

The observation that a GFlowNet's flow acts as a completion-feasibility signal is **not new**.
Nie et al. (arXiv:2605.07698, May 2026) name `Φ = Pr[valid completion | prefix]`, prove the correct
sampler is a Doob h-transform, and show local masking is exactly the `h ≡ 1` case. Ghari et al.
(NeurIPS 2024) already threshold a GFlowNet's flow to gate edits in biological sequence design.

What is not taken: the identity stated for GFlowNets, the fact that the oracle is *free* where every
competitor trains a second estimator, support loss in a non-prefix edit MDP, and exact
`|reachable| / |feasible|` measurement. Grammar-constrained decoding is **not** the right analogy —
it masks via an incremental parser, so its support is exact and its failure mode is probability
distortion, which is disjoint from the one described here.

---

## 16. Controls that were not run

* **`masked-screen@matched-proxy`** — deliberately superseded, see §7.
* **A slower-moving anchor.** Re-anchoring is greedy-to-best throughout. Stored traces show the
  greedy anchor takes essentially the maximum step available every round (17.50 of 21 on
  `protocol-alde`, staying put on 0.0% of rounds) and that jump size barely predicts the round's
  gain (Spearman ρ = 0.001). That is evidence of no *visible* headroom, not evidence that no better
  policy exists.
* **A GFlowNet on `large-space`** — see §2.
* **β against surrogate quality across tasks.** β varies on the selection landscape and nowhere
  else, so the cross-task contrast this would need does not exist in the store.
* **Batch-level saturation.** Older records store only per-round aggregates, so the marginal
  contribution of the k-th well to `E[max]` is not computable from them.
* **An external reproduction.** Everything compared here is this project's own artefact, except the
  holo audit — which is precisely why that audit carries the weight it does.

---

## How to cite anything on this site

Quote the task, the arm, the seed count and the statistic, and prefer W/T/L with a sign test to a
mean where §6 applies. Check the `deterministic` flag and, until the bless in §9 is recorded, check
which source fingerprint the record carries. Do not quote `cmaes+dp` or `mlde+earlyfit` as published
methods (§3). Do not quote a proxy-call saving as a budget saving (§5).
