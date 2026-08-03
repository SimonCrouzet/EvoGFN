# The benchmark suite

A benchmark is not a landscape and a number. It is a set of tests, each chosen because it can
settle a question the others cannot, run under a protocol a wet lab would recognise.

This page explains what a *task* is, what a *protocol* is, what each task in the suite
decides, and how to run the whole thing.

!!! warning "No results on this page"
    The suite is mid-change — search radius, anchoring rule and arm list have all moved — so
    every measured figure has been removed rather than refreshed. What each test is *for* is
    stable and is what you will find here. The numbers live in `results/`, one JSONL record
    per campaign, each carrying the fingerprint of the code that produced it. See [what this
    does not show](limitations.md) for which past claims were retracted and why.

---

## A protocol is three numbers, and their product is the only one a claim can be indexed by

```python
Protocol(rounds=4, batch_size=96, max_mutations=4, label="four plates")
```

`rounds × batch_size` is the **oracle budget**: how many variants get measured. A result
reported without its budget cannot be compared to one that has it, and the surveyed literature
routinely compares across budgets differing by two orders of magnitude.

### The grounding

| Campaign | Per round | Rounds | Total |
|---|---|---|---|
| EVOLVEpro (*Science* 2025) | 11–12 | 4–8 | 50–90 |
| LaMBO-2 wet lab | ~125 | 3 | 374 |
| ALDE (Arnold lab, *Nat Commun* 2025) | 216 / 90 / 90 | 3 | 396 |
| MLDE / ftMLDE (Wittmann 2021) | 384 + 96 | 2 | 480 |
| CLADE (Qiu & Wei 2021) | 96 | 5 | 480 |
| TrpB (Buller, *PNAS* 2015), classical | 528 / 1408 / 1144 | 3 | ~3,080 |

Against which the machine-learning convention — 10 rounds of 100 or 128, and 10,000 for
GFN-AL on AMP — sits above even *classical* directed evolution. The sharp version: MLDE's
entire claim is reaching the answer in ~480 assays instead of ~3,000, and a benchmark run at
10,000 has given that back before the first comparison is made.

`WET_LAB_PROTOCOLS` and `ML_CONVENTION` name the real ones, so an experiment can cite a
campaign rather than a round number someone liked.

!!! danger "The survey is a survey finding, and nothing more"
    The budget table above is sourced from primaries and stands on its own. The inference
    people want to draw from it — that benchmarking above the wet-lab regime *reverses
    conclusions* — is a separate empirical claim, and this project has no current measurement
    of it. The budget gradient that once addressed it ran under an anchoring rule that made
    every arm at every budget search the same ball, so it could not have shown a flip; it has
    not been re-run. Use the survey as a reason to *measure at the wet-lab budget*, and do not
    cite it — or us — as evidence that the ML budget produces wrong rankings.

### `max_mutations` sometimes does nothing, and it is worth checking

`Protocol.constrains_search(sequence_length)` returns `False` when the mutation budget reaches
every sequence anyway. On GB1 — four sites, four mutations — it is `False`. A result on GB1
therefore says nothing about search under a mutation constraint.

---

## A task is a landscape, a protocol, and a reason to run it

```python
Task(
    name=..., purpose=..., build=..., protocol=..., max_mutations=..., reanchor=..., attainable=...
)
```

The field a task cannot omit is `purpose`: what this task decides that the others do not. A
suite is only as good as its ability to distinguish methods, so a row that cannot say what it
settles should be deleted rather than kept for completeness. `build` is a factory rather than
an instance, so each seed can draw its own landscape where that is meaningful.

`reanchor` and `attainable` are keyword-only with **no defaults**, which is deliberate
enforcement rather than style. Both were once absent rather than false, and a default would let
the next task added inherit the same silence.

### The radius is per *round*, and the anchor moves

`max_mutations` bounds one round, not one campaign. Under `reanchor` the campaign moves its
parent to the best design measured so far at the end of each round, so `rounds × max_mutations`
substitutions accumulate from the wild type while the per-round radius stays at something a
round of site-saturation mutagenesis actually buys. `Task.search_budget` is that product;
without re-anchoring it is just `max_mutations`, because the extra rounds re-search the ball
the first one already covered.

This is not a detail. With a fixed anchor and a radius of 4, an Ehrlich instance's planted
optimum sits tens to hundreds of substitutions outside the search space — so it was not merely
hard to find, it was absent, and regret was being reported against a target no method could
reach.

### Regret is against what the search space was audited to contain

Even with the anchor moving, the reachable set is not the landscape. Each task declares what
`evogfn.benchmark.attainable` measured it to hold — the audit is run by
`experiments/audit_optima.py` and its answers are written down as constants, because
recomputing them on import would make the suite unusable — and a stored record's regret is
against **that**, not against the landscape's nominal optimum. Where the
audit could enumerate the reachable set the declaration is exact; where it could only bracket
it, the interval is carried rather than collapsed to a point, and regret is stored against the
conservative end — the only one witnessed by a design that was actually constructed.

A task with no audit declares nothing, and then **no regret is stored for it at all**. An
absent number is recoverable; a number measured against an unreachable target is not
distinguishable from a real one once it is in the column.

---

## Main tests: the rows that carry claims

| Task | Landscape | Protocol | Per round | Re-anchors | What it decides |
|---|---|---|---|---|---|
| `gb1-anchor` | GB1, 149,361 measured variants | 4 × 96 = 384 | 4 | no | Do the numbers hold on **real measurements**? The empirical anchor — and the easiest geometry here |
| `large-space` | Ehrlich `L=256, c=4, k=8, q=4` | 4 × 96 = 384 | 62 | yes | Can the method search a space it **cannot enumerate**? Stanton et al.'s own base configuration |
| `feasibility` | Ehrlich `L=64`, transition density 0.15 | 4 × 96 = 384 | 4 | no | Can the method stay **inside the constructible set**? Rejection sampling burns proposals where masking cannot |
| `protocol-alde` | Ehrlich `L=64`, density 0.5 | 3 × 132 = 396 | 21 | yes | Does the ranking survive the shape of a **real campaign**? After ALDE |
| `protocol-evolvepro` | same instance as above | 8 × 48 = 384 | 4 | yes | The **opposite shape** at a comparable budget, after EVOLVEpro. Many small rounds against few large ones |

Two rows do not re-anchor, and both for stated reasons. `gb1-anchor` has four sites and a
budget of four, so the first round already sees every design a later one could be anchored at.
`feasibility` holds its anchor still because what binds there is the transition matrix rather
than the radius — leaving the anchor fixed is what keeps its attainable optimum an *enumerated*
answer rather than the output of a search.

Sequence lengths follow published practice rather than convenience. Stanton et al.'s own base
configuration is `L = 256`; HDBO uses `L = 5, 15, 64` and reports two published Bayesian
optimisation methods running out of memory at 64. So the flagship large-space task uses
Stanton's base configuration (directly comparable to the benchmark's authors), the mid-size
tasks use `L = 64` (where the published field degrades), and diagnostics use `L = 32` (cheap
enough to sweep an axis at 50 seeds).

!!! warning "`gb1-anchor` is the easiest geometry in the suite"
    Four sites, no feasibility constraint, and a mutation budget that reaches every sequence —
    `Protocol.constrains_search` returns `False`, and there is nothing for a rejection-sampling
    control to reject. GB1 says the numbers are not an artefact of synthetic landscapes. It
    says nothing about constrained search, and an earlier version of this project claimed
    otherwise.

!!! warning "An audited task can be saturable as well as winnable"
    Four of the five main tasks are audited to contain their nominal optimum, which is what
    makes them winnable — and also what makes them exhaustible. **A task an arm has already
    solved cannot rank the arms above it**, whatever its seed count. The report marks such arms
    `SOLVED` and labels comparisons drawn on them vacuous rather than printing a p-value about
    a ceiling.

---

## Diagnostics: the rows that inform choices

Diagnostics vary one axis on a fixed, cheap `L = 32` landscape at 50 seeds. They inform
things — whether the ranking survives a change of budget, whether rounds matter at fixed
total, how much room there is between the training objectives. They are how choices get
informed, not what gets claimed, and `Purpose.DIAGNOSTIC` is what keeps that in the type.

| Diagnostic | Tasks | Axis varied |
|---|---|---|
| `budget_gradient()` | `budget-96`, `budget-384`, `budget-1000`, `budget-10000` | oracle budget, from the wet-lab regime to the ML convention |
| `rounds_curve(budget)` | `rounds-8x48`, `rounds-4x96` | many small rounds against few large, at a fixed total |
| `objective_task()` | `objectives` | GFlowNet training objective, at equal budget |

The budget and rounds diagnostics were both run under the old fixed-anchor regime, which made
every arm in each sweep search the identical ball — so whatever they showed was not about
budget or about rounds. Both now re-anchor and neither has been re-run. The objective
comparison is superseded by the [selection phase](selection.md), which runs it at a seed count
the diagnostic's own power estimate asked for.

---

## The methodologies

A methodology is whatever turns a task and a seed into a runnable campaign. Keeping that one
callable is what makes a GFlowNet variant, a classical baseline and a baseline-with-model-
access the same kind of thing to the harness, so no arm can quietly receive a different budget,
surrogate or starting point than another.

!!! info "The arm list is in flux"
    The suite is mid-transition from "every classical baseline silently receives a surrogate"
    to "every classical baseline is the pipeline its paper describes, with one named ablation
    ladder". The **principle** below is settled; the exact set of arm names is not, and this
    table will move under you. Read the arms out of `evogfn.benchmark.methods` — `BASELINES`,
    `OBJECTIVES` and `flow_objectives()` — rather than out of this page.

| Arm | What it is |
|---|---|
| `random` | the floor: mutate at random inside the budget |
| `hill-climb` | neighbours of the incumbent, restarting after a patience window — HDBO's greedy incumbent search, *not* the wet-lab DE walk |
| `single-step` | traditional directed evolution to ALDE's specification: saturate one site, fix its best residue, never revisit it |
| `recomb` | Li et al.'s other DE arm: saturate every site independently on one background, then combine the winners |
| `genetic` | the Ehrlich paper's own algorithm, at its own rates — the reference every arm is paired against |
| `genetic-feasible` | a GA that rejection-samples until its offspring are legal — the feasibility control |
| `cmaes` | CMA-ES over a continuous relaxation, as published |
| `adalead` | FLEXS' recommended benchmark algorithm — evolutionary search whose rollout screens every candidate against its own surrogate |
| `mlde` | machine-learning-directed evolution — what protein engineers actually run, at almost exactly this budget |
| `alde` | its active-learning successor, at the configuration its authors took to the bench: one-hot encodings, a five-member bootstrapped ensemble, Thompson sampling |
| `gfn-tb`, `gfn-contrastive`, `genetic-gfn` | GFlowNet objectives |
| `gfn-db`, `gfn-subtb`, `gfn-fldb` | the detailed-balance family, which needs a policy with a flow head |

Two arms in that list are the two different things the field calls directed evolution, and keeping
them apart is deliberate: `hill-climb` draws a random single substitution anywhere in the sequence
and keeps no record of which positions it has changed, while `single-step` exhausts one position
before committing to it and then never touches it again. Every "MLDE beats DE" claim in the wet-lab
literature is measured against the second.

**Simulated annealing is not an arm.** It appears in no baseline table of either lineage this suite
is read against, so it is nobody's expected comparator. `SimulatedAnnealing` remains importable from
`evogfn.algorithms.baselines` for anyone who wants the comparison; what was removed is the results
row, not the sampler.

**The two DE arms cost a fraction of the campaign, and the surplus goes to replicates.** Li et al.
cost `single-step` at `19 x n_site + 1` samples and `recomb` at `19 x n_site + 2` — one plate's worth
against a campaign of four. Neither source says what to do with the difference, because in their
setting there is none: they report each DE arm at its own fixed cost. **Our choice** is to run
several copies of the protocol at different site orders and pool them into each plate, and to start
another when they all finish with budget left. That is the same protocol run again rather than a new
mechanism — Li et al.'s own simulation enumerates every site order and every starting variant — and
it is the direction a forced deviation in a baseline should run: filling the plate with repeats
instead would handicap the arm by most of its budget.

Two consequences for reading such a row. It is a **best over site orders at the campaign's budget**,
where the sources report an **average over site orders at one walk's cost**; a comparison against a
published DE number has to say which it is. And on a library whose sites are the whole sequence —
`gb1-anchor` — the replicates cannot differ, the pooled request deduplicates back to the protocol's
own cost, and the surplus shows up as duplicate wells. That is the honest answer there rather than a
gap. Read the DE arms' real spend off `requested`, not off the oracle-call column.

**GFlowNets train against a proxy, never the oracle.** Each builds a `ProxyLandscape` over the
same surrogate instance the campaign refits, so training costs proxy evaluations and never
oracle calls. Charging them would exhaust a 384-call campaign before the first round finished,
and no published method does it. The surrogate is *constitutive* of that pipeline — it is what
makes a GFlowNet trainable at 384 assays at all — rather than an extra it is being handed.

---

## Methods are compared as published

**Hyperparameters are the ones each method's own authors chose.** The genetic algorithm runs at
the Ehrlich paper's mutation and recombination rates, MLDE in the regime Wittmann et al. report.
Where a value comes from a paper, the source is named in the code beside it rather than being a
number someone liked.

**A baseline gets no component its paper does not describe.** No deep ensemble screening its
pool, no proxy to optimise against, and a candidate pool its own paper would recognise — a GA's
pool is its population, CMA-ES's is `lambda`, MLDE's is an exhaustive library because that is
its protocol. Pool size is part of the method, not a harness setting; a single global pool
could not be right for more than one of them.

**Anything beyond the published pipeline is its own named arm.** The attribution question a
reviewer will ask — was it the surrogate or the constructive sampler? — is real, and it is
answered by a ladder on one representative baseline rather than by a silent default on all of
them. Each rung adds exactly one thing to the rung above it:

| Rung | What it adds |
|---|---|
| `genetic` | nothing; the model does not exist |
| `genetic+screen` | the model filters the pool; the search itself stays blind |
| `genetic+search` | the sampler also optimises against the model |
| `genetic+distinct` | the plate is filled with distinct designs rather than with proposals |

`random+screen` is the same first rung on the floor, which is what says whether a screen helps
at all or only helps a method that was already searching.

These are **decomposition rows, not controls**, and the report labels them as such on every
line they appear on. A reader who takes one for the yardstick is reading exactly the comparison
this arrangement exists to prevent: the reference is `genetic`, a published pipeline, because a
pipeline is what a lab actually chooses between.

That leaves exactly one method in the table with no published settings to inherit: ours. A
GFlowNet run at defaults against a tuned field is not being compared to it — the comparison
measures our configuration, in the direction that flatters the baselines. So the GFlowNet's
training objective, reward exponent and gradient steps are chosen by a
**[selection phase](selection.md)** that runs before the benchmark, under a rule written down
before any of its numbers existed, on a diagnostic landscape no headline task uses. Its answer
is an input to these tables rather than one of their rows.

---

## Tiers, and which of them may be quoted

Tasks are grouped into tiers, and each tier carries a `Purpose` that decides what its results
are allowed to be used for. A single "is this the headline" flag could not express this: a
diagnostic *measures how methods behave*, while a selection tier measures nothing at all — it
chooses **our own** configuration, and a choice made on a landscape a claim is later drawn from
is tuning on the test set.

| Tier | Purpose | Carries claims | Tasks |
|---|---|---|---|
| `main` | `BENCHMARK` | yes | `gb1-anchor`, `feasibility`, `protocol-alde`, `protocol-evolvepro` |
| `large-space` | `BENCHMARK` | yes | `large-space`, run separately at fewer seeds because a campaign at `L = 256` costs an order of magnitude more |
| `objectives` | `DIAGNOSTIC` | no | the objective comparison |
| `rounds-curve` | `DIAGNOSTIC` | no | `rounds-8x48`, `rounds-4x96` |
| `budget-gradient` | `DIAGNOSTIC` | no | the four budget tasks |
| `sensitivity` | `SELECTION` | no | one GFlowNet setting moved at a time, on the diagnostic landscape |

`Tier.headline` is `True` for exactly the `BENCHMARK` rows. That is the property a results
table should filter on, and it is why the distinction lives in a type rather than in a comment.

---

## Running it

```bash
uv run python experiments/select_configuration.py       # first: fix the GFlowNet's configuration
uv run python experiments/run_suite.py                  # everything
uv run python experiments/run_suite.py --tier main      # headline only
uv run python experiments/run_suite.py --task feasibility --method genetic   # one cell
uv run python experiments/run_suite.py --seeds 200      # main-tier seed count (default 100)
uv run python experiments/run_suite.py --report         # no runs, just read
```

Safe to interrupt and safe to re-run. Every campaign is written to `results/` the moment it
finishes, and a second invocation runs only what is missing — so raising a tier's seed count
from 30 to 50 costs twenty campaigns per arm, not fifty.

Sharding is by process, not by thread: `--task` and `--method` each narrow the run, the store
keeps one file per task and method so writers never collide, and every campaign is seeded from
its own seed rather than from process order. A sharded run and a serial one therefore produce
identical records. Raising the thread count instead would not — see the reproducibility note
below.

The headline tiers read `results/selected.json` for the GFlowNet arm they should run. If the
selection phase has not been run, they fall back to the untuned defaults and say so, rather than
reporting them as though they had been chosen. **No selection has been recorded at present**, so
that is the state you will get.

### Staleness

A stored result carries a fingerprint of the code that could have produced it: one hash per
`.py` file, over the transitive import closure of declared entry points
(`evogfn.benchmark.methods` and `evogfn.loop.campaign`). Declaring the entry points rather than
deriving them is what makes the mechanism pay — a record goes stale only when something it can
actually reach has changed. Hashing the whole package tree instead invalidated thousands of
campaigns that no edit could have influenced, and the difference between "correct" and "usable"
here is exactly that.

A stale record is re-run rather than trusted: `--report` prints how many seeds each task and
method holds and which of them are stale, along with the module names that changed.

---

## How to read a result

The suite reports a **paired comparison**: the same seeds, the same task, the same protocol,
so a difference is a difference between methods. Three numbers, and all three are needed.

* **The mean advantage** and its confidence interval — what to expect on average.
* **The win/tie/loss count** — what a single laboratory campaign should expect. A method can
  win comfortably in expectation while a lab running one campaign loses a substantial share of
  the time, and reporting the mean alone hides exactly that.
* **The number of seeds.** 100 for the main tiers, 30 for `large-space` (campaign cost differs
  by an order of magnitude between `L = 4` and `L = 256`, not because the claims differ), 50
  for diagnostics.

Beside them the report prints **proxy spend**, because it is a budget someone chose rather than
a constant of an architecture — `steps × batch_size` for the GFlowNet, `generations ×
population` for the `genetic+search` ablation. An arm that wins on regret while spending an
order of magnitude more surrogate evaluations has won on compute, and this column is the only
place a reader would see it.

!!! warning "Determinism is enforced for torch, and only for torch"
    A multithreaded matmul sums its partial products in thread-completion order, and a few
    hundred gradient steps amplify that into a different design. `configure_determinism()` pins
    it, and both experiment scripts exit with code 3 rather than running unpinned. But what
    actually binds is `torch.set_num_threads`: the same function's `os.environ.setdefault` on
    `OMP_NUM_THREADS` and friends is a **no-op**, because those are read by the BLAS at import
    and numpy has already been imported by the time it runs. `is_deterministic()` reads
    `torch.get_num_threads()` alone, so a record's `deterministic: true` is a statement about
    torch and nothing else. Pinning the BLAS would take a launcher or a `sitecustomize`, not a
    call in `main()`. See [§7 of the limitations](limitations.md).

---

## API

- [`evogfn.benchmark`](reference/benchmark.md) — tasks, protocols, suite, harness, store.
- [Choosing the configuration](selection.md) — the selection phase, and how to rerun it.
- [What this does not show](limitations.md) — the full ledger of claims and their status.
