---
date: 2026-08-10
topic: EvoGFN re-framing after the amortisation null; and the next direction
verdict: PURSUE
nugget: "Constrained sequence-design benchmarks score methods against sets those methods cannot construct — and once you score against the reachable set, every construction-respecting mechanism ties, so the differences the literature reports are differences in support."
---

# Evaluation: EvoGFN after the amortisation null

Session ran the full Seed → Diverge → Evaluate → Deepen → Frame → Decide pipeline:
three brainstormers (re-frame / scout-next / steelman-the-null) and three idea-critics
(one per starred framing). Every number below was recomputed from live `results/`
JSONL during the session, not taken from notes.

## Verdict: PURSUE — framing A, with C as its mechanism section

Retitle onto **off-support**. The 99/100 finding on Stanton et al.'s own generator is
the only asset that is simultaneously surprising, about someone else's code, already at
n=100, and independent of whether the GFlowNet is any good. Framing C's ties stop being
an embarrassing null and become A's *prediction*: score against the reachable set and
all construction-respecting mechanisms are optimising over the same small object, so of
course they tie. Framing B (the controls) goes to a non-archival workshop.

**This decision was pre-registered.** The 2026-08-04 evaluation's revisit condition read:
"Fall back to the measurement-first framing — attainability, support and repair share as
the headline, with the method as the instrument that made them measurable — if the
transfer probe returns null." It returned null on 2026-08-09. The contingency has fired.

## Dimension scores

| Dimension | A — Off-support | B — The controls | C — Mechanism factorial |
|---|---|---|---|
| Novelty | Months (6–12), unoccupied | Weeks–Months; genre well-trodden | Weeks–Months; taxonomy is Coello Coello 2002 |
| Impact | Medium, High if a ranking moves | Medium, scoped | Medium; Low as "mechanism beats search" |
| Timing | Well-timed, seam closing | Well-timed, closing | Well-timed, closing |
| Feasibility | High risk — no draft exists | High risk on writing, not compute | High risk as scoped; low if rescoped |
| Competition | Open — needs holo + edit action space + audit instinct | Moderate — Prescient could run it | Open |
| Nugget | Fuzzy: "four incarnations" is a list | Fuzzy: two nulls welded to a conclusion | Fuzzy: the 10× is a detection limit |
| Narrative | Workable, strongest arc | Scolding; hardest to sell | Null-result arc |
| **Verdict** | **REFINE → PURSUE as spine** | **REFINE → workshop** | **REFINE → section 5 of A** |

## What was measured this session (all from disk, no new runs)

- **`cmaes+dp` ties or nearly ties the GFlowNet and beats it on diversity.** Paired,
  100 seeds: best +0.0081 [−0.0223,+0.0385] on protocol-alde, +0.0369 [+0.0072,+0.0666]
  on protocol-evolvepro, +0.0000 on feasibility. Diversity −0.9126, −1.4879, −0.8832,
  all CIs excluding zero. Nonparametric (immune to the reward lattice): W/T/L 32/43/25
  p=0.43 on alde; 39/42/19 **p=0.012** on evolvepro.
- **Amortisation null is robust.** gfn vs `+reinit`: 23/52/25 p=0.885 and 25/53/22
  p=0.771.
- **Compute gap 72–202×, and `cmaes+dp` uses NO surrogate.** GFlowNet 332.8/300.3/161.5
  CPU-s with 38,400/134,400/57,600 proxy calls; `cmaes+dp` 3.09/1.49/2.24 CPU-s with
  **0** proxy calls.
- **The surrogate is noise on exactly the tasks where the headline win lives.**
  Pearson r(prediction, oracle) on the selected batch: protocol-alde +0.060/+0.073;
  protocol-evolvepro +0.052→+0.040; gb1-anchor +0.175→+0.473; trpb-anchor +0.181→+0.470;
  feasibility +0.015→+0.942. (Range-restricted by greedy selection; the cross-task
  contrast is the signal.)
- **Round 0 — an untrained, near-uniform masked draw — beats every published baseline's
  entire campaign on protocol-alde**: 0.3463 vs genetic 0.2437 final, hill-climb 0.2662
  final, random 0.1688 final.
- **The dissociation.** Where the constraint binds (Ehrlich protocol tasks): masking
  does the work, surrogate is noise, GFlowNet's round-0→final gain (+0.228) ≈
  `cmaes+dp`'s (+0.232). Where it does not (GB1, no constraint): GFlowNet starts
  *behind* (2.64 vs genetic 3.75) and posts the largest gain of any arm (+3.93).
  **Surrogate quality predicts which term dominates.**
- **The forward mask is local, not sound.** `_substitution_mask` checks only that the
  resulting state keeps every adjacency permitted; it never asks whether a feasible
  completion exists. It is the locally-constrained-decoding approximation, which is
  what produces the support gap.
- **`+terminal` collapses**: 0 feasible of 2,048 proposals per round, best −inf. This is
  the rebuttal to "just don't mask intermediates."
- **`genetic-feasible` fails 100/100** on `feasibility`, exhausted at 96 of 384.

## Key concerns

1. **The 99/100 is a property of edit-from-anchor action spaces, not of holo.** Resolved
   analytically: under left-to-right de novo generation the planted optimum is reachable
   by construction, since each prefix extension only needs the newly-formed pair and the
   target is feasible. So the honest claim is that Ehrlich is sound for whole-sequence
   optimisers and unsound for edit-based methods — which is most of what it is used to
   benchmark. State this scope first, before a reviewer does.
2. **Pseudo-replication.** protocol-alde and protocol-evolvepro are the same Ehrlich
   instance; `feasibility` is a constant (0.375 on 100/100 for every construction-
   respecting arm); GB1 has no constraint. Effective n on landscapes is ~1.
   `cmaes+dp` is not run on any `replicate-*-i{3,5,7}`.
3. **The reward lattice.** `best` takes 4 values on protocol-alde, step 0.0625, larger
   than every within-cluster effect. Report sign tests and W/T/L, not means. Do not
   print a "10×" ratio — it is a signal-to-detection-limit ratio, not an effect ratio.
4. **The budget strand cannot carry weight.** `budget-*` has no GFlowNet arms at all and
   `budget-10000` has two arms populated. Demote to a motivating paragraph.
5. **Everything in the comparison is Simon's own artefact.** One external reproduction is
   the highest-leverage week available.
6. **`docs/limitations.md` is systematically stale** — verified wrong on proxy_calls
   (reads 57,600, not 0), TrpB (has a task), rejection sampling on `feasibility`
   (fails 100/100, does not solve it), and genetic-gfn's rank. Do not cite it.
7. **No draft exists.** `manuscript/` holds two files and there is no `.tex`. Prose, not
   compute, is the binding constraint.

## Watch list

- Prescient Design (Stanton, Frey) — own holo/Ehrlich, best positioned to notice.
- Mila GFlowNet group (Jain, Malkin, Bengio) — amortisation argument is theirs.
- KAIST (Kim, Ahn) — Genetic-GFN, δ-CS.
- SynFlowNet / RxnFlow / RGFN — **not threatened; say so explicitly.** Their constraint
  is a path property, so reachability is their specification. Framing this as a partition
  rather than an indictment converts three hostile reviewers into supportive ones.

## The method thesis (the "problem to solve")

Added after the verdict, on the question of where a *method* contribution still lives.

The paper's scoping law generates the method problem rather than closing it off.
`_substitution_mask` is **local**: it guarantees the next state is feasible and never
asks whether a feasible completion exists. That is the locally-constrained-decoding
approximation, and it is unsound in the same way hard grammar masking is unsound for
LLMs (Grammar-Aligned Decoding, Park et al., NeurIPS 2024; ASAp corrects it, but only
where prefix oracles are tractable).

So:

> Where the feasibility predicate **factorises**, an exact projection computes the
> completion oracle and a learned constructive sampler has nothing left to buy — which
> is exactly why the GFlowNet ties `cmaes+dp`. Where it **does not factorise**, no exact
> projection exists and no sound mask exists, and an amortised learner is the only thing
> that can approximate the completion-feasibility oracle.

That is a problem where the GFlowNet is necessary rather than ceremonial. It reuses
`gfn-fldb` (forward-looking DB is already the right shape — point the forward-looking
estimate at feasibility rather than reward), and it is evaluable exactly here and
essentially nowhere else, because `reachable_terminal_states` gives the true support and
enumerable instances give exact L1.

**Fail-fast, days, no campaigns:** on an enumerable instance, measure |feasible| and
|reachable| under a factorising predicate versus a non-factorising one, and check whether
the gap widens. Do not use "forbidden token pair anywhere" as the non-factorising case —
it is secretly a DP over seen-token subsets.

### Candidate angles — all tentative, none started

Ranked by whether a real problem sits underneath, and whether it needs a GFlowNet
*specifically* rather than any sampler. Nothing here has been run; the fail-fast column
is what would decide each one. Three of the tests are free — analysis over records
already stored.

**A1. Completion-oracle / non-factorising constraints.** As above. The learned sampler
approximates the completion-feasibility oracle where no exact projection exists.
*Fail-fast (days, no campaigns):* on an enumerable instance, measure |feasible| vs
|reachable| under a factorising predicate versus a non-factorising one, and check whether
the gap widens. Do **not** use "forbidden token pair anywhere" — secretly a DP over
seen-token subsets.
*Why it might fail:* a cheap approximate projection may exist for most predicates people
actually care about, leaving the learned sampler with a thin margin again.

**A2. The batch is the object, not the sample.** A lab orders a plate of 96; a GFlowNet
draws 96 i.i.d. samples. The value of a plate is `E[max]` over the set, so the training
objective and the decision object differ. No paper in this literature conditions on the
partial batch. This also dissolves the diversity problem honestly — mean pairwise Hamming
is a proxy for a quantity that can be computed directly, which is why a non-learned
method beat the GFlowNet on it (3/3 tasks, CIs excluding zero).
*Fail-fast (free):* from stored per-round records, compute the marginal contribution of
the k-th batch member to `E[max]`. If it saturates at k≈10 of 96, most of every plate is
wasted and the gap is real and sized.
*Why it might fail:* if `E[max]` does not saturate, the i.i.d. batch is already close to
optimal and there is nothing to fix.

**A3. Re-anchoring as a learned decision.** Currently greedy-to-best. But the holo port
shows re-anchoring is what makes planted optima reachable at all, so the anchor trajectory
is the load-bearing object, and choosing it is a sequential decision under uncertainty
that no GA makes and no DP addresses. It is also the one place amortisation could pay,
because the decision recurs.
*Fail-fast (free):* `anchor_distance` is recorded every round and has never been
analysed. Check whether greedy re-anchoring is ever beaten by a slower-moving anchor in
the stored traces.
*Why it might fail:* greedy may simply be near-optimal at 4–8 rounds; the decision
horizon is short.

**A4. Masked construction under an *uncertain* mask.** The current constraint is a known
transition matrix. Real feasibility is "does it fold, does it express" — predicted, not
given. A mask built from an uncertain classifier compounds error along the path, and
false negatives permanently delete regions of the space. Probably the angle a wet-lab
reader cares about most.
*Fail-fast (cheap, existing machinery):* corrupt the known transition matrix at rate ε
and measure how the reachable set and campaign performance degrade.
*Why it might fail:* it may reduce to "your classifier should be well-calibrated," which
is not a GFlowNet contribution.

**A5. Training against a reward that is noise, non-stationary, and self-generated.**
Surrogate–oracle r ≈ 0.05 on the Ehrlich protocol tasks, refit each round from the
policy's own samples, with 300 gradient steps taken against it. That feedback loop is
unstudied. The method response is an adaptive temperature.
*Fail-fast (free, and the one to run first):* both halves are already on disk — a β sweep
from 0.1 to 100 in `objectives`, and per-task surrogate correlations. Does the optimal β
track surrogate quality across tasks? If it does, "β should adapt to how much the proxy
can be trusted" is a mechanism *and* a method, from stored records, and it slots into the
current paper rather than competing with it.
*Why it might fail:* β is confounded with landscape scale across tasks, so the comparison
may not be clean with only a handful of tasks.

**A6. Preference-conditioned multi-objective.** The one setting where the amortisation
multiplier exceeds 1 — a non-amortised method must re-optimise per preference vector,
while a conditioned policy answers all of them from one training run. `cmaes+dp` has no
clean multi-objective analogue, so the instrument that hurts elsewhere cannot follow here.
*Fail-fast:* restructure `mo-preferences` so the *number of preferences* is the swept
axis, against a scalarised baseline re-run per preference at matched total compute.
*Why it might fail:* it is a replication of Jain et al. (MOGFN, ICML 2023) in a new
regime; the regime has to carry the novelty.

**A7. Library-as-action.** Change the object proposed: a degenerate-codon library
specification rather than 96 individual sequences, with an oversampling cost model. The
lab's real primitive. A product-form distribution over sequences is what a factorised
forward policy is already shaped to fit.
*Fail-fast:* add the cost model and re-score existing stored batches by what they would
cost to order; check whether the ranking inverts under cost.
*Why it might fail:* needs a landscape with more candidate sites than a human would
enumerate — invisible on GB1, where the library is the whole space (ledger A2).

**A8. Path multiplicity and `P_B`.** The subset lattice makes `log P_B = −log k!` exact,
and Deleu et al. (UAI 2024) show MaxEnt RL is biased under path multiplicity. So
GFlowNet vs soft RL on this lattice is a clean, GFlowNet-specific technical claim that is
currently unmeasured (ledger E1/E2). More theory-flavoured than the rest.
*Fail-fast:* run PPO/soft-Q on the identical MDP and check whether the mutation-count
distribution shifts as predicted.
*Why it might fail:* the bias may be real but too small to cost fitness at this budget.

## Revisit conditions

- If `untrained-masked` and `masked-screen@matched-proxy` do **not** return by 24 Aug
  2026, drop ICLR 2027 and target NeurIPS 2027 D&B (~June) rather than writing into the
  deadline.
- If those controls come back mixed across the two protocol tasks (the most likely single
  outcome, given `+reinit` already flipped sign between them), that is the worst branch
  for a fixed deadline — skip the venue rather than write the alternative paper.
- If `large-space` (L=256, never run with a GFlowNet) shows the GFlowNet's margin over
  `cmaes+dp` *growing* with L, the null becomes a scoping result and the GFlowNet
  premise is restored. Scout at 20 seeds only if the answer can land by 1 Sept.
- Next-direction candidates evaluated but not started: support-correct diversity
  re-measurement (days), and library-as-action / degenerate-codon design (~4 weeks) as
  the strongest surviving candidate for a *method* thesis.
