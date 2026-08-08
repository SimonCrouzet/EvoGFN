# The benchmark suite

A benchmark is not a landscape and a number. It is a set of tests, each chosen because it can
settle a question the others cannot, run under a protocol a wet lab would recognise.

This page explains what a *task* is, what a *protocol* is, what each task in the suite
decides, which tier decides what, and how to re-run any of it.

!!! warning "No results on this page"
    The suite is mid-change — search radius, anchoring rule and arm list have all moved — so
    every measured figure has been removed rather than refreshed. What each test is *for* is
    stable and is what you will find here. Configuration lives here: seed counts, budgets, plate
    sizes, task parameters. Measurements do not — the numbers live in `results/`, one JSONL record
    per campaign, each carrying the fingerprint of the code that produced it, and a figure copied
    onto this page would go stale silently. See [what this does not show](limitations.md) for which
    past claims were retracted and why.

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
every sequence anyway. On GB1 and on TrpB — four sites, four mutations — it is `False`. A result on
either empirical anchor therefore says nothing about search under a mutation constraint.

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
| `trpb-anchor` | TrpB, 159,129 of 160,000 variants measured | 4 × 96 = 384 | 4 | no | Does the empirical result survive a **second protein**? Johnston et al.'s four active-site positions of tryptophan synthase — same shape as GB1, a different assay and far less gradient |
| `large-space` | Ehrlich `L=256, c=4, k=8, q=4` | 4 × 96 = 384 | 62 | yes | Can the method search a space it **cannot enumerate**? Stanton et al.'s own base configuration |
| `feasibility` | Ehrlich `L=64`, transition density 0.15 | 4 × 96 = 384 | 4 | no | Can the method stay **inside the constructible set**? Rejection sampling burns proposals where masking cannot |
| `protocol-alde` | Ehrlich `L=64`, density 0.5 | 3 × 132 = 396 | 21 | yes | Does the ranking survive the shape of a **real campaign**? After ALDE |
| `protocol-evolvepro` | same instance as above | 8 × 48 = 384 | 4 | yes | The **opposite shape** at a comparable budget, after EVOLVEpro. Many small rounds against few large ones |

Three rows do not re-anchor, and all three for stated reasons. `gb1-anchor` and `trpb-anchor`
each have four measured sites and a budget of four, so the ball of radius four *is* the whole
library and the first round already sees every design a later one could be anchored at — a
property checked per landscape rather than inherited, and one that would fail the moment a
task's sites outnumbered its budget. `feasibility` holds its anchor still because what binds
there is the transition matrix rather than the radius — leaving the anchor fixed is what keeps
its attainable optimum an *enumerated* answer rather than the output of a search.

`trpb-anchor` is the second empirical anchor rather than a duplicate of the first. Both are four
coupled positions over twenty amino acids and near-complete, which is what makes them
comparable; the assay, the protein and the reward geometry differ, which is what makes running
both worth the compute. Its audit is its own — an enumeration of all 160,000 reachable terminal
states from `VFVS` at four mutations — and comes back exact at **2.4505**, the landscape's own
optimum, so its regret floor is genuinely zero. Nothing is borrowed from GB1's declaration.

Sequence lengths follow published practice rather than convenience. Stanton et al.'s own base
configuration is `L = 256`; HDBO uses `L = 5, 15, 64` and reports two published Bayesian
optimisation methods running out of memory at 64. So the flagship large-space task uses
Stanton's base configuration (directly comparable to the benchmark's authors), the mid-size
tasks use `L = 64` (where the published field degrades), and diagnostics use `L = 32` (cheap
enough to sweep an axis at 50 seeds).

!!! warning "The two empirical anchors are the easiest geometry in the suite"
    Four sites, no feasibility constraint, and a mutation budget that reaches every sequence —
    `Protocol.constrains_search` returns `False` on both, and there is nothing for a
    rejection-sampling control to reject. GB1 and TrpB say the numbers are not an artefact of
    synthetic landscapes. They say nothing about constrained search, and an earlier version of
    this project claimed otherwise.

!!! warning "An audited task can be saturable as well as winnable"
    Five of the six main tasks are audited to contain their nominal optimum, which is what
    makes them winnable — and also what makes them exhaustible. **A task an arm has already
    solved cannot rank the arms above it**, whatever its seed count. The report marks such arms
    `SOLVED` and labels comparisons drawn on them vacuous rather than printing a p-value about
    a ceiling.

### Replicates: the same protocol on other landscape draws

The two protocol tasks deliberately share one Ehrlich instance so that only campaign *shape*
differs between them — which is what makes that comparison clean, and which leaves every
constrained result in the suite resting on a single draw from the generator. A hundred seeds vary
the wild type and the surrogate's initialisation; not one of them varies the landscape. "Does
this ordering hold on a different instance" is therefore a question the headline tables cannot
answer at any seed count.

`replication()` answers it by rebuilding both shapes on further draws. Everything but the
generator's seed is held at the headline task's own settings — length, alphabet, motif structure,
density, protocol, anchor rule and attainable declaration — so a difference between a replicate
and its headline task is a difference between instances and nothing else.

| Task | Shape | Draw |
|---|---|---|
| `replicate-alde-i3`, `replicate-alde-i5`, `replicate-alde-i7` | 3 × 132 = 396, radius 21, re-anchored | `REPLICATION_SEEDS = (3, 5, 7)` |
| `replicate-evolvepro-i3`, `replicate-evolvepro-i5`, `replicate-evolvepro-i7` | 8 × 48 = 384, radius 4, re-anchored | the same three |

The attainable declaration transfers for a stated reason rather than by resemblance: the bound is
an argument about the *budget* — that this many re-anchored rounds at this radius reach the
budget-split bound — and budget, length, radius and round count are all held fixed. The draw
decides where the optimum sits, not whether the protocol can walk far enough to reach one.

!!! warning "Three draws is an assumption, and the report says so"
    Three was chosen a priori and has **not** been shown to be enough. It is already known to be
    too few in one respect: three draws cannot clear
    [`unanimity_floor`][evogfn.benchmark.statistics.unanimity_floor] — a unanimous verdict over
    three reaches only sign-test *p* = 0.250 — so no seed count makes an ordering significant with
    the instance as the unit. The report prints `UNDERPOWERED BY DESIGN` on every such family.
    `experiments/variance_pilot.py` is what measures how many draws are actually needed.

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
| `constraint_density()` | `density-0.05`, `density-0.15`, `density-0.25`, `objectives`, `density-1` | how much of the sequence space is constructible at all |
| `anchor_study()` | `objectives`, `anchor-fixed` | whether the anchor moves, and whether the policy comes with it |

The budget and rounds diagnostics were both run under the old fixed-anchor regime, which made
every arm in each sweep search the identical ball — so whatever they showed was not about
budget or about rounds. Both now re-anchor and neither has been re-run. The objective
comparison is superseded by the [selection phase](selection.md), which runs it at a seed count
the diagnostic's own power estimate asked for.

### The constructibility sweep is read on a different column

Masking actions does not restrict a sampler to the feasible set. It restricts it to the feasible
states reachable *through feasible intermediates*, which is smaller — a design can satisfy the
transition constraint, sit inside the mutation budget, and still have no construction order in
which every intermediate is legal. `unconstructible_fraction` measures that gap on designs a run
actually produced, and on the headline tasks it is telemetry rather than a result: one number, at
one density, with nothing to read it against.

`CONSTRAINT_DENSITIES = (0.05, 0.15, 0.25, 0.5, 1.0)` turns it into a curve. Each rung is the
shared diagnostic instance in every parameter but the density, so a share that moves across the
family moves with the constraint and nothing else. What each rung is for: **1.0** permits every
adjacency and is the axis's own control, since a non-zero share there is a fault in the
measurement rather than a property of a landscape; **0.5** is `DIAGNOSTIC_DENSITY`, and that rung
**is** `objective_task` returned rather than copied, so the curve passes through the very
campaigns the other diagnostics are read at; **0.15** is what the headline `feasibility` task
runs at; **0.25** keeps those two from being adjacent points; **0.05** is about as sparse as this
vocabulary can be while the constraint still discriminates between designs.

Two consequences a reader has to hold on to. Only an arm that **breeds** has a share to report —
the quantity counts offspring a genetic teacher produced and the policy could not construct — so
the tier names its four arms explicitly (`DENSITY_ARMS`) rather than taking the default set:
`genetic-gfn` because it is the only arm that breeds, and `genetic`, `genetic-feasible` and
`gfn-tb` because they are what the share has to be read against. And every rung but the shared
one **declares no attainable optimum**: lowering the density shrinks the reachable set, so the
audited value at 0.5 is not theirs, and carrying it across would put an unclearable floor into
every regret on the sparser rungs. This family is read on the constructibility columns, not on
regret.

### The anchor study is a list of cells, not a cross

Two orthogonal mechanisms, and conflating them answers a different question from the one the
project's claim rests on. **Does moving the ball help?** — a property of the protocol, available
to every method, which is why a genetic algorithm is on that axis: if re-anchoring lifts it too,
the protocol is doing the work. **Given that it moves, does bringing the trained policy along
help?** — that is amortisation, and it is the thing a learned constructive sampler has that a GA
structurally cannot, since a GA's operator is the same before and after a move.

| Cell | Anchor | Policy |
|---|---|---|
| `objectives` / `gfn-tb` | moved | carried |
| `objectives` / `gfn-tb-rebuilt` | moved | rebuilt |
| `anchor-fixed` / `gfn-tb` | fixed | no state to carry |
| `objectives` / `genetic` | moved | none |
| `anchor-fixed` / `genetic` | fixed | none |

Five cells, not six. The cross of two tasks with three arms also contains a *rebuilt* policy on
the task whose anchor never moves — and since nothing is ever rebuilt there, that is the carried
arm's campaign under a second name. `run_anchor_study` therefore runs the study cell by cell, and
`--method` is refused inside this tier because an arm filter would silently drop whole cells.
`anchor-fixed` declares no attainable optimum, deliberately: `DIAGNOSTIC_ATTAINABLE` is what four
*re-anchored* rounds reach, and storing regret against it here would report the difference between
two reachable sets as the arm's shortfall.

---

## The methodologies

A methodology is whatever turns a task and a seed into a runnable campaign. Keeping that one
callable is what makes a GFlowNet variant, a classical baseline and a baseline-with-model-
access the same kind of thing to the harness, so no arm can quietly receive a different budget,
surrogate or starting point than another.

!!! info "Read the arms out of the code, not out of this page"
    The **principle** below is settled and the names are current as of this writing, but the set
    moves. `evogfn.benchmark.methods` is the authority — `BASELINES`, `OBJECTIVES`,
    `flow_objectives()`, `variant_arms()` and `DECODER_STUDY` — and
    `uv run python experiments/run_suite.py --tier <name> --report` prints what a given tier
    actually resolves to.

| Arm | What it is |
|---|---|
| `random` | the floor: mutate at random inside the budget |
| `hill-climb` | neighbours of the incumbent, restarting after a patience window — HDBO's greedy incumbent search, *not* the wet-lab DE walk |
| `single-step` | traditional directed evolution to ALDE's specification: saturate one site, fix its best residue, never revisit it |
| `recomb` | Li et al.'s other DE arm: saturate every site independently on one background, then combine the winners |
| `genetic` | the Ehrlich paper's own algorithm, at its own rates — the reference every arm is paired against |
| `genetic-feasible` | a GA that rejection-samples until its offspring are legal — the feasibility control |
| `cmaes` | CMA-ES over a continuous relaxation, as published, decoding greedily |
| `adalead` | FLEXS' recommended benchmark algorithm — evolutionary search whose rollout screens every candidate against its own surrogate |
| `mlde` | machine-learning-directed evolution — what protein engineers actually run, compressed to this suite's shared 384 |
| `mlde-over-budget` | the same method at the budget its own paper spends: 384 screened plus one designed plate. **Over budget on every task in the suite**, and here because it does not fit |
| `mlde+earlyfit` | **ours**: MLDE at a training size a constrained screen can actually return. Not Wittmann et al.'s protocol |
| `alde` | MLDE's active-learning successor, at the configuration its authors took to the bench: one-hot encodings, a five-member bootstrapped ensemble, Thompson sampling |
| `random+screen`, `genetic+screen`, `genetic+search`, `genetic+distinct` | the decomposition ladder, below |
| `gfn-tb`, `gfn-contrastive`, `genetic-gfn` | GFlowNet objectives |
| `gfn-db`, `gfn-subtb`, `gfn-fldb` | the detailed-balance family, which needs a policy with a flow head |

**Two MLDE rows are deviations, and both are marked.** `mlde-over-budget` spends one plate more
than every other arm, which is exactly what its paper's protocol is: a *training size*, read off
the protocol as "every plate but the last screens, the last is designed", so on this suite's
four-plate tasks it resolves to Wittmann et al.'s own 384 + 96 by construction rather than by two
constants agreeing. It is in the table *because* it does not fit — a headline resting on the
compressed row alone would rest on a comparator we had shortened to a quarter of its training
set. Reading its row needs two caveats: the stored `protocol` field is the task's, so it names 384
while the arm spent 480 (the gap shows in `oracle_calls`, and `extra_rounds` in the arm's own
`parameters` says where it came from); and on a re-anchoring task the extra plate is extra *reach*
as well as extra assays, so its regret is taken against a bound audited for a shorter campaign —
read its `best` rather than its `regret` where the two disagree.

`mlde+earlyfit` is the other direction: a *smaller* method than the one it sits beside. Both
published MLDE arms gate their handover on **usable** measurements, and `MLDE.observe` discards an
infeasible assay because there is no fitness to regress on — so on a landscape where most wells
come back with nothing, the supervised phase is unreachable at any budget a laboratory would run,
and both rows spend the whole campaign screening at random. Both stay, because that is the
finding. This arm lowers the training size so the handover happens at the same oracle budget as
everyone else, which is how *would MLDE be competitive here if it ever got to be MLDE?* can be
asked at all. The value is derived from this suite's own measured feasible share and appears
nowhere in Wittmann et al., so the row must never be quoted as MLDE — the `+` is the disclosure.

**`cmaes+dp` is deliberately not a baseline.** A separable Gaussian over per-position logits
cannot represent a transition constraint, so on a constrained instance the raw decode is not
buildable and something has to repair it — and which repair is chosen is a choice *we* make on the
baseline's behalf, specified nowhere in the literature. The reported `cmaes` arm takes the obvious
one: accept substitutions by descending gain while the design stays legal. The exact alternative is
a dynamic program that returns the best legal design outright; it scores better, and that is
precisely why it lives in `DECODER_STUDY` rather than `BASELINES`. It is a stronger decoder than
any published account of the method uses, and a baseline reported through machinery of our own
devising is no longer the published method. It is kept beside the reported arm so the difference
between them can be measured, under the same `+` the ablation rungs wear.

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
`gb1-anchor` and `trpb-anchor` — the replicates cannot differ, the pooled request deduplicates back
to the protocol's own cost, and the surplus shows up as duplicate wells. That is the honest answer
there rather than a gap. Read the DE arms' real spend off the record's `proposals` and
`duplicate_fraction`, not off the oracle-call column.

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

### The GFlowNet has a ladder of its own, and a separate step to ship a rung

`variant_arms()` crosses two GFlowNet mechanisms on the configuration the project ships. Writing
`B` for the shipped arm's name, which the ladder resolves from `results/selected.json` rather than
restating:

| Rung | What it adds |
|---|---|
| `B` | nothing; feasibility is masked at every intermediate |
| `B+terminal` | feasibility is required of the *design*, not of the construction order |
| `B+anchor` | the policy is told which parent it is evolving from |
| `B+terminal+anchor` | both |
| `B+wide` | neither; only the capacity that `+anchor` also brought |

Four rungs and not three, because unlike the `genetic+` ladder these two mechanisms are
**orthogonal** rather than nested: one changes the graph the policy walks, the other changes what
the policy reads at each state. Three arms would report two main effects and leave the interaction
unmeasured, which is exactly where the interesting answer could be — and the both-on rung is the
configuration the method would ship if both rungs win, so it has to have been *run* rather than
inferred by addition.

`B+wide` is a **capacity control, not a step**. Anchor conditioning widens the trunk's input — the
state embedding, the anchor embedding and one difference indicator per position — so `+anchor` is
both a conditioning change and a capacity change, and a win there is unattributable until a plain
policy of the same size has been measured. Its width is resolved by `matched_hidden_dim`, which
*builds candidates and counts their parameters* rather than evaluating a closed form; at the
shipped trunk width of 64 it comes out at 101. No integer width matches exactly, so the rule is
the narrowest plain trunk that is **not smaller** than the conditioned one, which puts the residual
on the safe side: a control given slightly more capacity can only understate the mechanism it
exists to isolate. It carries the base arm's prefix because its width is derived from the base
configuration and would match no other — under a bare name, a record sized against one selection
and a record sized against another would be the same `(task, arm)` cell.

The base is what ships, not `gfn-tb`, and that is not cosmetic: the selection study measured the
reward exponent on trajectory balance and then again on sub-trajectory balance, and **the curve
reversed direction between them**. An effect measured on one objective is therefore not known to
transfer to the other here, so a ladder built on `gfn-tb` would be a study of an arm nobody runs.

!!! warning "A rung reaches the headline table only through `--promote`"
    The ladder runs on the diagnostic landscape under `Purpose.SELECTION`. Its rungs do **not**
    appear in `main` because the tier exists; they appear because somebody ran

    ```bash
    uv run python experiments/run_suite.py --promote RUNG
    ```

    which runs nothing, reads the ladder's already-banked campaigns, prints every rung against the
    base, and writes `results/promoted.json`. `methods_for` reads that file and nothing else, so
    until it exists the headline tiers run exactly what they ran before the ladder was built.

    The ordering is the whole point. If the rungs sat in `main` and the best were picked
    afterwards, the configuration would have been chosen on the tasks that carry the claims —
    tuning on the test set. The rung is **named by the caller and never derived**, because
    deriving the winner would make promotion an automatic consequence of the ladder having run.
    What the step checks is that the evidence permits the named rung: a non-base rung must have
    beaten the base, paired across shared seeds, with the interval excluding zero; the base itself
    may be promoted — "neither mechanism earns its compute" is a real outcome — but only when no
    other rung beat it. A refusal exits non-zero rather than warning.

    Once promoted, all five rungs enter the default path: the promoted one as the method, the rest
    as decomposition rows labelled `[ablation of ...]` exactly where `genetic+search` is, each
    carrying an attribution line beside its p-value. It is not free — five GFlowNet arms in place
    of one is four extra campaigns per task per seed — and that cost is paid only by somebody who
    both promotes and then runs.

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

Tiers run in the order below — cheap and decisive first, so an interrupted night still yields
something readable.

| Tier | Purpose | Seeds | Tasks | Arms |
|---|---|---|---|---|
| `objectives` | `DIAGNOSTIC` | 50 | `objectives` | `OBJECTIVES` plus the flow-head family; GFlowNet-only, since a classical baseline has no objective to vary |
| `sensitivity` | `SELECTION` | 50 | `objectives` | nine arms, one setting moved at a time |
| `variant-ladder` | `SELECTION` | **100** | `objectives` | the five rungs of `variant_arms()` |
| `main` | `BENCHMARK` | 100 | `gb1-anchor`, `trpb-anchor`, `feasibility`, `protocol-alde`, `protocol-evolvepro` | every baseline plus the shipped GFlowNet arm |
| `replication` | `BENCHMARK` | 100 | the six replicate tasks | the same arms as `main` |
| `rounds-curve` | `DIAGNOSTIC` | 50 | `rounds-8x48`, `rounds-4x96` | as `main`, less `mlde-over-budget` |
| `budget-gradient` | `DIAGNOSTIC` | 50 | the four budget tasks | as `main`, less `mlde-over-budget` |
| `constraint-density` | `DIAGNOSTIC` | 50 | the five density rungs | `DENSITY_ARMS`: `genetic`, `genetic-feasible`, `gfn-tb`, `genetic-gfn` |
| `anchor-study` | `DIAGNOSTIC` | 50 | `objectives`, `anchor-fixed` | `gfn-tb`, `gfn-tb-rebuilt`, `genetic` — as named cells, not a cross |
| `large-space` | `BENCHMARK` | **30** | `large-space` | as `main` |

`Tier.headline` is `True` for exactly the `BENCHMARK` rows. That is the property a results
table should filter on, and it is why the distinction lives in a type rather than in a comment.

### Why each seed count is what it is

**100 for the headline tiers, 30 for `large-space`, 50 for the diagnostics** — the split is about
campaign cost, which differs by an order of magnitude between `L = 4` and `L = 256`, not about the
claims differing.

**`replication` is seeded like `main` because it answers a question about `main`.** A
lower-powered replicate could not distinguish "the ordering broke" from "we ran out of seeds",
which is the one thing it exists to decide.

**`variant-ladder` is seeded like the headline despite being a selection tier**, because the rung
that justifies the fourth arm is an *interaction*: `+terminal+anchor` earns its compute only by
beating what the two single rungs predict by addition, and a difference of differences carries
about twice the standard error of either main effect. At the diagnostic count that rung comes back
inconclusive against its own prediction — the one answer this tier must not return.

**`constraint-density` and `anchor-study` both take the diagnostic count, and specifically the
`objectives` count**, because in each the pivotal cell *is* an `objectives` campaign: the rung at
`DIAGNOSTIC_DENSITY` and the moved-and-carried anchor cell are both `(objectives, gfn-tb)`, already
stored at that count. Asking for another would leave the control powered differently from
everything it controls, and would put a second estimate of the same measurement in a second table
for a reader to reconcile. Nothing argues for more on the density family in particular: four of
its five rungs declare no attainable optimum, so seeds bought for a regret comparison would buy
precision on a column the tier does not report.

**`sensitivity` shares the objectives task deliberately** — same landscape, same protocol, same
seeds — so a setting's effect and an objective's are measured against each other rather than across
two configurations. It is `SELECTION` and not `DIAGNOSTIC` because both its rungs are off by
default: what it returns *chooses the configuration the method ships* rather than describing how
methods behave, and called a diagnostic it would be eligible to appear in a results table as a
mechanism finding while the same campaigns had already fixed our own configuration.

### One arm is removed from two tiers, by name

`mlde-over-budget` spends a plate beyond its task's protocol. That is the point of the arm, and it
is exactly what a tier whose **axis is the budget** cannot hold constant: on `rounds-curve` it
breaks the fixed total the whole curve is defined by, and on `budget-gradient` it sits one plate to
the right of the point it is plotted at — a doubling at the `8×12` rung and a 1% perturbation at
the `10×1000` one. The same arm would be a different distortion at every rung, in the direction
that flatters it, and the curve would read as a property of the methods. So `BUDGET_AXIS_TIERS`
drops it, and it stays in `main`, `replication` and `large-space`, where the budget is held fixed
and the extra plate is the whole point.

---

## Running it

```bash
uv run python experiments/select_configuration.py       # first: fix the GFlowNet's configuration
uv run python experiments/run_suite.py                  # everything
uv run python experiments/run_suite.py --tier main      # headline only
uv run python experiments/run_suite.py --task feasibility --method genetic   # one cell
uv run python experiments/run_suite.py --seeds 200      # main-tier seed count (default 100)
uv run python experiments/run_suite.py --diagnostic-seeds 100   # diagnostics (default 50)
uv run python experiments/run_suite.py --seed-from 0 --seed-to 25   # one seed shard
uv run python experiments/run_suite.py --results results-scratch   # write elsewhere
uv run python experiments/run_suite.py --report         # no runs, just read
uv run python experiments/run_suite.py --promote RUNG   # ship a ladder rung, and stop
```

Safe to interrupt and safe to re-run. Every campaign is written to `results/` the moment it
finishes — one JSONL file per task and method, at `results/<task>/<method>.jsonl` — and a second
invocation runs only what is missing, so raising a tier's seed count from 30 to 50 costs twenty
campaigns per arm, not fifty.

Sharding is by process, not by thread. `--task`, `--method` and the `--seed-from` / `--seed-to`
range each narrow the run; the store keeps one file per task and method so writers never collide;
and every campaign is seeded from its own seed rather than from process order. A sharded run and a
serial one therefore produce identical records. Raising the thread count instead would not — see
the reproducibility note below. Two details worth knowing: a seed shard still *reports* the tier's
full seed set, so a shard says what the store holds rather than what that process was handed; and
`--method` is refused inside `anchor-study`, which runs named `(task, arm)` cells rather than a
cross, because an arm filter there would silently drop whole cells.

`--report` and `--promote` run no campaigns. Everything else refuses to start unless threading is
pinned, exiting with code 3.

The headline tiers read `results/selected.json` for the GFlowNet arm they should run, and
`results/promoted.json` for the ladder rung. A selection **has** been recorded — the shipped arm is
`gfn-subtb@b0.1-s300-l0.9-h64`, and it replaces the untuned GFlowNet arms rather than joining them
— while **no promotion has**, so the ladder's rungs do not appear in the headline tiers. If either
file is absent the suite falls back and says so, rather than reporting untuned defaults as though
they had been chosen; if either is *unfinished*, it raises rather than falling back, because a
partial record describes a configuration no rule ever chose. Check the files themselves rather than
this paragraph, and `--report` will print the arm names a tier actually resolves to.

### The other entry points

Each of these is runnable on its own. The first three write campaigns into `results/` through the
same store; the transfer probe and the multi-objective suite use roots of their own; the last three
run analyses rather than campaigns and write nothing under `results/`.

| Command | What it does |
|---|---|
| `experiments/select_configuration.py` | the [selection phase](selection.md): fixes the GFlowNet's objective, reward exponent, gradient steps and width before the benchmark runs. Writes `results/selected.json` |
| `experiments/variance_pilot.py` | sizes the instance resampling: how many landscape draws `replication` needs, from a measured between-instance variance rather than an assumed one. `--report` reads without running |
| `experiments/proxy_saturation.py` | the proxy-budget ladders, so the `proxy` column reports a measured saturation point rather than a budget somebody picked. `--ladder steps` / `--ladder generations`, `--report` |
| `experiments/transfer_probe.py` | the transfer probe: train at one anchor, move to another, measure the first frozen plate before any new learning. `--level near` / `--level far`, `--report`. Writes to `results-transfer/`, deliberately a separate root |
| `experiments/run_multi_objective.py` | the multi-objective suite, below. Writes to `results-mo/` |
| `experiments/audit_optima.py` | re-derives what each task's search space can reach, and puts regret against the nominal optimum beside regret against the attainable one. Writes nothing under `results/` |
| `experiments/distributional_fidelity.py` | the exact L1 between the trained policy and $p^*(x) \propto R(x)^\beta$ on an enumerable instance — the one metric an optimiser that never samples cannot satisfy |
| `experiments/feasible_reachable_sweep.py` | what masking excludes: feasible designs with no feasible construction order, swept over transition density |

`src/evogfn/benchmark/holo_port.py` is a **library rather than a script**: it runs this package's
own procedures against holo-bench's Ehrlich instances, so "our generator is ours" is answered by
measurement rather than by argument. It compares the two generators' instance distributions, checks
that the two implementations agree on reward and feasibility *exactly* (this is the one comparison
where a disagreement is a bug), runs the unmodified attainability audit on holo's landscapes, and
measures both CMA-ES decoders' repair rate on them. Drive it from a short snippet over
`REFERENCE_SHAPES` and `sweep_shape`; the write-up lives in `notes/holo-bench-port.md`.

### The multi-objective suite is a separate run

`experiments/run_multi_objective.py` has its own tasks, arms, tiers and store root (`results-mo/`),
and its stored `best` and `regret` mean different things — hypervolume above the campaign's
reference point, and IGD+ against its reference front. Its tiers are `preferences`
(`DIAGNOSTIC`, and the only one that decides anything: how many preference vectors the main-table
arm gets), `main` (`BENCHMARK`), and the `conflict` and `objectives` sweeps (`DIAGNOSTIC`).
Defaults are 50 seeds for `main` and 30 for the rest, adjustable with `--seeds` and
`--explanatory-seeds`; `--report` reads without running.

Its arm registry is laid out to match `BASELINES` arm for arm where the two suites can share an
arm at all — `random`, `genetic`, `random+screen`, `genetic+screen`, `genetic+search` — plus
`nsga2` and two GFlowNet rows. Note the naming: what used to be `gfn-tb` here is now
**`gfn-tb-scalar`**, because `mogfn-pc` was registered beside it. `gfn-tb-scalar` is GFlowNet-AL
over a **fixed** weighted-sum scalarisation; `mogfn-pc` samples a preference per step and
conditions the policy on it. A row named plain `gfn-tb` two lines above one named `mogfn-pc` reads
as a stronger claim than the arm supports, so the name now carries half the distinction and
`SCOPE_NOTES` prints the rest beside it. The rename landed while nothing was stored under either
key — every record is keyed by the arm name, so afterwards it would have orphaned the campaigns.
What is *missing* against the single-objective list is missing on purpose: `hill-climb`, `cmaes`,
`mlde`, `alde`, `adalead`, `single-step`, `recomb` and `genetic-feasible` are single-objective
pipelines whose published forms say nothing about a front.

### Staleness

A stored result carries a fingerprint of the code that could have produced it: one hash per
`.py` file, over the transitive import closure of four declared entry points
(`RESULT_DEPENDENCIES`). Declaring the entry points rather than deriving them is what makes the
mechanism pay — a record goes stale only when something it can actually reach has changed. Hashing
the whole package tree instead invalidated thousands of campaigns that no edit could have
influenced, and the difference between "correct" and "usable" here is exactly that.

| Entry point | Why it is declared |
|---|---|
| `evogfn.benchmark.methods` | how a campaign is built: every sampler, surrogate, acquisition rule and landscape hangs off it |
| `evogfn.loop.campaign` | how it is run, and the ledger and metrics that come with it |
| `evogfn.benchmark.suite` | the task definitions, the per-task mutation budgets, and the line that writes `proxy_calls` into the record — none of it reachable by import from the two above |
| `evogfn.benchmark.selection` | the swept arms and their hyperparameters, so an edit there changes what a swept arm *is* |

The rule this encodes is that a result is invalid if it would change under the current code, and
valid if it would not. Hashing the import closure is a *conservative approximation* of that: it
cannot know whether an edit alters an outcome without re-running, so it assumes the worst. Where a
change provably cannot alter a completed run,
[`ResultStore.bless`][evogfn.benchmark.store.ResultStore.bless] is the escape hatch — and it
requires naming the modules being vouched for, precisely because a blanket restamp is dozens of
independent assertions made in one call. A blessed record stays distinguishable from a re-run one:
`RunRecord.blessed` records which names were asserted rather than measured.

A stale record is re-run rather than trusted: `store.usable()` drops it, so it vanishes from the
report's tables until it is regenerated, and `--report` prints how many seeds each task and method
holds, which of them are stale, and the module names that changed.

---

## How to read a result

The suite reports a **paired comparison**: the same seeds, the same task, the same protocol,
so a difference is a difference between methods. Every arm is compared against one reference, and
which one depends on the tier — a tier that does not contain the default gets its own, because a
missing reference makes the whole paired section collapse into one line, which reads as "nothing
separated these arms" when what happened is that nothing was tested.

| Tier | Reference | Why |
|---|---|---|
| everything not listed below | `genetic` | the Ehrlich paper's own algorithm at its own hyperparameters — a **published pipeline**, because a pipeline is what a lab chooses between. Pairing against `genetic+search` would pair every headline number against a hybrid we invented |
| `objectives` | `gfn-tb` | trajectory balance, the objective the others are alternatives to |
| `sensitivity` | `steps-300` | the shipped setting, so each swept value reads as a change from what the headline rows were produced at |
| `variant-ladder` | the ladder's own base rung | resolved from the recorded selection rather than named, so it cannot become an arm the tier stopped running |
| `anchor-study` | `gfn-tb` | so the printed pair on the re-anchored task is rebuilt-against-carried: the amortisation cell, and the only cell a paired test can reach here. The other axis — moved against fixed — is across two *tasks*, and is read off the two tables rather than tested |

Three numbers, and all three are needed.

* **The mean advantage** and its confidence interval — what to expect on average.
* **The win/tie/loss count** — what a single laboratory campaign should expect. A method can
  win comfortably in expectation while a lab running one campaign loses a substantial share of
  the time, and reporting the mean alone hides exactly that.
* **The number of seeds.** 100 for the main tiers, 30 for `large-space` (campaign cost differs
  by an order of magnitude between `L = 4` and `L = 256`, not because the claims differ), 50
  for diagnostics.

Beside them the report prints **proxy spend**, because it is a budget someone chose rather than
a constant of an architecture — roughly `steps × batch_size` for the GFlowNet, `generations ×
population` for the `genetic+search` ablation. An arm that wins on regret while spending an
order of magnitude more surrogate evaluations has won on compute, and this column is the only
place a reader would see it. It is **read off the record** rather than derived from those closed
forms, and the difference is not cosmetic: the formula describes an arm that only samples from its
own policy, and it is wrong for an arm that *breeds*, whose teacher's proposals are themselves
proxy evaluations the formula never counts. One caveat the column carries: a sampler rebuilt at an
anchor move restarts its own accounting, so on a re-anchoring task the stored count covers the last
anchor's rounds and reads as a floor.

### Four things a mean cannot say, and where the report says them

A row of averages hides the cases where the arm was not the thing its name claims. Each of these
gets a marker on the row itself, not a footnote, because an unmarked row is the one that gets
quoted.

* **`exhausted=n`** — campaigns that could not finish, because the sampler could not propose
  designs it had not already measured. Their `best`, `diversity` and `feasible_fraction` are `nan`
  and their `regret` is `None`, never zero, and they are excluded from every mean — so the count
  has to appear beside it or the row reports a subset while `n` reads like a full one. An arm that
  exhausted on *every* seed gets a line of its own rather than a row of `nan`s. Storing nothing at
  all is what this replaced: the arm then simply vanished, and an empty cell is an absence a reader
  may fill in as they please, including as the sharpest result on the table.
* **`unfit=n` and a `NEVER FITTED` line** — a supervised arm whose model never fitted spends its
  budget, fills every plate and reports a regret arithmetically indistinguishable from a fitted
  one's. Every paired comparison naming such an arm is marked as a difference against a *random
  screen wearing a supervised method's name*. Reading such a row as a tuned method that lost fairly
  is the one conclusion the column exists to prevent.
* **`SOLVED`** — an arm sitting on the attainable optimum on at least half its seeds
  (`VACUOUS_SHARE`). Comparisons naming it are labelled vacuous before the p-value is read, because
  significance against a ceiling is significance about the task.
* **`[ablation of ...]`** — a decomposition row, on the table line *and* beside every p-value it
  appears in, with a sentence naming which attribution question it answers: a classical ablation
  splits a surrogate's contribution from a sampler's, a promoted ladder rung splits one GFlowNet
  mechanism from another.

Where a comparison cannot be drawn it is said rather than skipped. An omitted line reads as "these
did not separate", which is the opposite of what happened; so a tier with no reference arm, a
reference with no usable seeds, and a pair whose two arms completed different numbers of campaigns
each print a line saying so. An inconclusive comparison prints the seed count that would resolve it.

### Fields a stored record carries beyond the two it is indexed by

`RunRecord` has one larger-is-better field (`best`) and one smaller-is-better field (`regret`), and
every table reads them positionally. The rest of the record is what makes a surprising number
diagnosable without re-running the campaign:

| Field | What it says |
|---|---|
| `fitted` | **tri-valued.** `None` means the arm fits nothing — a genetic algorithm has no model whose state this could describe, and a share of nothing is not `False`. `False` is a measurement, and the one that matters: the arm ran its whole campaign in its random-screening stage. Records written before the field existed load as `None`, so the default cannot retroactively accuse them |
| `exhausted` | the campaign failed to finish. Its `rounds`, `oracle_calls` and `proposals` still describe the part that *did* run, so the record says how far it got |
| `repaired_fraction` | for an arm decoding a continuous relaxation, the share of designs that reached the plate only through a repair. An **attribution** field, not a performance one: at 0.0 the search distribution found the constructible set unaided; at 1.0 every design credited to the method was chosen by the repair subject to the method's preferences |
| `bred_designs` | designs a genetic teacher produced and the policy was asked to construct a path to. Zero for every arm that breeds nothing, which is every arm except Genetic-GFN — and it is what makes `unconstructible_fraction` readable, since a share of nothing is also zero |
| `draws_attempted`, `draws_rejected`, `draws_unmutated` | the rejection-sampling counters, which travel together or not at all. Two are a rate and the third is what the rate *means*: the mutation kernel runs at `p_m = 1/L`, so roughly a third of every batch carries no substitution at all, and those trivially-legal draws hold the rejection rate down to something that reads survivable while the arm makes no progress |
| `duplicate_fraction` | the share of a round's proposals that repeated something else in that same round. Duplicates are charged, so this is budget spent re-measuring the method's own convergence. Within-round only |
| `cpu_seconds`, `wall_seconds` | processor time is **the comparable cost figure**; elapsed time is context only, since a sharded suite contends for cores by a factor that says nothing about any method |
| `parameters` | the arm's resolved configuration, one scalar per setting. Provenance only — nothing compares it, and no value of it can make a record stale. It exists because a setting passed in from outside a closure is invisible to the fingerprint, which used to leave the arm's *name* as the only record of what it ran at |
| `rounds` | one dict per round: what it proposed, screened, measured and found, plus `anchor_distance`. Flat at zero there says the campaign searched one Hamming ball for its whole life |

### Replicated tasks are read one instance at a time, and refuse otherwise

The per-task tables are already per instance. What a reader does with them is the risk: averaging
six tasks × 100 seeds into one figure of 600 does not look like a mistake from the outside — it
looks like *more* evidence, at a standard error several times narrower than the design supports.
Instance × seed pairs are not independent observations, because a seed varies the wild type and the
surrogate's initialisation *within* a draw and nothing in the pooled array varies the draw.

So the report adds a section that takes the **draw** as the unit: each arm's per-draw effect, the
interval across draws at `n = draws`, and a sign test on how many draws agreed — the one
across-instance statistic that assumes nothing about how the effect is distributed, which at three
or four draws is the only honest thing to quote. Beside every arm it prints the measured factor by
which pooling would have understated the standard error, so the refusal comes with its own
arithmetic.

!!! danger "`pooled_metric()` raises rather than being merely absent"
    On a task set holding two landscape draws of one protocol shape, `pooled_metric()` **raises**.
    That function exists to be the door somebody walks through when they need a single number for a
    slide, and to shut in exactly that case. An absent function is simply re-written as
    `np.concatenate` at the call site, and a convention is what the next person needing one figure
    breaks. `instance_effects()` is what to use instead.

!!! warning "Determinism is enforced for torch, and only for torch"
    A multithreaded matmul sums its partial products in thread-completion order, and a few
    hundred gradient steps amplify that into a different design. `configure_determinism()` pins
    it, and every experiment script exits with code 3 rather than running unpinned. But what
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
