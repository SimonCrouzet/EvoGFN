# EvoGFN

**EvoGFN** is a Python library for in-silico directed evolution with Generative Flow Networks
(GFlowNets). It generates batches of sequence variants that are diverse and high-fitness at the same
time, rather than many near-copies of a single best hit.

Use it to run directed-evolution campaigns against a fitness landscape — your own, or one of the
built-in benchmarks — and to compare a GFlowNet against classical baselines (traditional directed
evolution, MLDE, ALDE, AdaLead, a genetic algorithm, hill climbing, CMA-ES) on equal terms.

> [!NOTE]
> Early development. The API is not stable, and the benchmark suite is mid-change: the search radius,
> the anchoring rule and the arm list have all moved recently. This README and the documentation
> describe **design, protocol and how to reproduce** — they carry no measured figures. Results live
> in `results/`, where each record carries a fingerprint of the code that produced it.

---

## Why you might want this

A design round has a fixed budget — you can synthesise and assay so many variants and no more. If your
proposal method returns 96 sequences that are minor variations on one design, you have spent the
budget on a single bet. Anything that kills that design kills the whole round.

Optimisers do this by construction: they climb toward one peak. A GFlowNet learns a policy that samples
variants *in proportion to* their predicted fitness, so a batch spreads across the high-fitness regions
of the landscape instead of piling onto one. Applied to mutation trajectories from a parent sequence,
that is a direct model of what directed evolution actually does.

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SimonCrouzet/EvoGFN
cd EvoGFN
uv sync                  # GPU (CUDA build of torch)
uv sync --extra cpu      # CPU only — 1.1GB instead of 4.7GB
```

Everything in the default path runs on CPU. A GPU helps for long sequences and for the optional
protein-language-model oracles (`--extra plm`).

Run a single campaign to check the install:

```bash
uv run evogfn campaign landscape=ehrlich
```

`landscape=ehrlich` is closed-form and needs no network. See
[`docs/getting-started.md`](docs/getting-started.md) for the Hydra override surface and the same
thing in Python.

---

## Built-in landscapes

You can plug in any fitness function by implementing one interface. Two are included, chosen because
something is known about their correct answers — which means you can check whether a method actually
worked, not just whether it produced a plausible-looking number.

| Landscape | What it is | Why it is useful |
|---|---|---|
| **Ehrlich** | Closed-form, procedurally generated sequences with tunable epistasis, ruggedness and feasibility constraints | A sequence scoring 1.0 is planted by construction, so the target is known rather than assumed. No download; evaluation is instant |
| **GB1** | Real deep-mutational-scanning data: 149,361 measured variants across 4 positions | Near-complete over a 160,000-sequence space, so regret is exact against a real assay rather than against the best thing anyone happened to find |

Two caveats that matter more than they look:

* **A planted optimum is not automatically a reachable one.** Whether a campaign's search space
  contains the answer depends on its mutation budget, its number of rounds and whether it re-anchors.
  That is audited per task rather than assumed, and regret is measured against the audited value.
  Getting this wrong once made three quarters of a reported regret column a constant nobody had
  computed.
* **GB1 is not combinatorially complete.** 10,639 of the 160,000 combinations were never assayed and
  are imputed as zero by default; `is_measured` exists so an analysis can exclude them instead.

Ehrlich landscapes also define which sequences are *constructible* at all. EvoGFN enforces this by
masking invalid actions during generation, so every proposed sequence is buildable by construction
rather than filtered out afterwards.

---

## The benchmark: what to run, and what each tier is for

Three kinds of run, and only one of them may be quoted. The distinction lives in a type — every
`Task` carries a `purpose` string saying what it decides that the others cannot, and every `Tier`
carries a `Purpose` of `BENCHMARK`, `DIAGNOSTIC` or `SELECTION`.

| Tier | Purpose | Carries claims | What it is for |
|---|---|---|---|
| `main` | `BENCHMARK` | **yes** | The headline: `gb1-anchor`, `feasibility`, `protocol-alde`, `protocol-evolvepro`, at 100 seeds |
| `large-space` | `BENCHMARK` | **yes** | The same, split out: a campaign at `L = 256` costs an order of magnitude more, so it runs at 30 seeds |
| `objectives` | `DIAGNOSTIC` | no | GFlowNet training objectives at equal budget, on a cheap `L = 32` landscape |
| `rounds-curve` | `DIAGNOSTIC` | no | Many small rounds against few large, at a fixed total |
| `budget-gradient` | `DIAGNOSTIC` | no | Oracle budget from the wet-lab regime to the ML convention |
| `sensitivity` | `SELECTION` | no | One GFlowNet hyperparameter moved at a time |

`Tier.headline` is `True` for exactly the `BENCHMARK` rows. Diagnostics explain behaviour and never
appear in a results table; a selection tier measures nothing at all — it chooses *our own*
configuration, and it must sit on a landscape no headline task uses.

### The selection phase runs first

Every classical baseline here runs at hyperparameters its own authors tuned — the genetic algorithm
at the Ehrlich paper's mutation and recombination rates, MLDE in the regime Wittmann et al. report. A
GFlowNet run at inherited defaults against that field is not being compared to it: the table would be
measuring our configuration, in the direction that flatters the baselines.

So the GFlowNet's configuration is *chosen*, by a rule fixed before any of its numbers existed —
lowest mean regret, with batch diversity breaking statistical ties — on a diagnostic landscape no
headline task uses. Three sequential stages: training objective, then reward exponent, then gradient
steps.

```bash
uv run python experiments/select_configuration.py                       # all three stages
uv run python experiments/select_configuration.py --stage a             # one stage
uv run python experiments/select_configuration.py --stage a --only gfn-tb   # one arm: the sharding knob
uv run python experiments/select_configuration.py --print-winner        # stage A's answer, for a coordinator
uv run python experiments/select_configuration.py --seeds 100           # seeds an arm must hold before a stage may choose
uv run python experiments/select_configuration.py --report              # read the store, run nothing
```

The phase writes `results/selected.json`. The benchmark **reads** that file rather than re-deriving
the choice, so a moved seed count or arm list cannot silently swap the arm a table reports.
**No selection is currently recorded** — the previous one was made under a superseded rule and has
been deleted — so the suite will fall back to the untuned defaults and say so. Full procedure, and
what a staged design cannot see, in [`docs/selection.md`](docs/selection.md).

### Then the suite

```bash
uv run python experiments/run_suite.py                       # everything
uv run python experiments/run_suite.py --tier main           # one tier
uv run python experiments/run_suite.py --task feasibility    # one task
uv run python experiments/run_suite.py --method genetic      # one arm
uv run python experiments/run_suite.py --seeds 200           # main-tier seed count (default 100)
uv run python experiments/run_suite.py --diagnostic-seeds 80 # diagnostic seed count (default 50)
uv run python experiments/run_suite.py --results DIR         # where the store lives (default results/)
uv run python experiments/run_suite.py --report              # read the store, run nothing
```

`--tier`, `--task` and `--method` are all repeatable.

**Everything is resumable.** Each campaign is appended to the store the moment it finishes, keyed by
`(task, method, seed)`, so a run killed at hour six keeps everything up to hour six and a rerun
computes only what is missing. Raising a tier from 30 seeds to 50 costs twenty campaigns per arm, not
fifty.

**Everything shards, one process per arm.** `--task` and `--method` are the knobs. This is safe
because campaigns are independent, the store keeps one file per task and method so writers never
collide, and every campaign is seeded from its own seed rather than from process order — a sharded run
and a serial one produce identical records. Raising the thread count instead would *not*: a
multithreaded reduction sums in completion order, and both scripts refuse to run unpinned (exit
code 3) for that reason.

```bash
for arm in random genetic hill-climb single-step recomb mlde alde adalead cmaes; do
  uv run python experiments/run_suite.py --tier main --method "$arm" &
done
wait
```

---

## Where results live, and how to tell a stale record from a usable one

Results land under `results/<task>/<method>.jsonl`, one JSON record per campaign, appended. Each
record carries what it measured (`best`, `regret`, `diversity`, `feasible_fraction`, `oracle_calls`,
`proxy_calls`, `cpu_seconds`, per-round detail, the best ten designs) plus two provenance fields:

* **`protocol`** — the *task's* repr, not the protocol's: name, rounds × batch, per-round radius and
  whether the anchor moved. Two records at `4x96=384` that differ in search radius are not comparable,
  and one naming only the budget could not be told apart from one that searched a space thirty times
  larger.
* **`source`** — a content hash per `.py` file, over the transitive import closure of the campaign's
  declared entry points (`evogfn.benchmark.methods` and `evogfn.loop.campaign`). Declaring entry
  points rather than hashing the package tree is what makes the mechanism usable: a record goes stale
  only when something it could actually reach has changed.

**A record is stale when any module in its own `source` map hashes differently now.** You do not have
to check this by hand:

```bash
uv run python experiments/run_suite.py --report
```

prints, per task and method, how many seeds are held, how many are stale, and which module names
changed. A stale record is silently excluded from every table and re-run on the next invocation
rather than trusted — `ResultStore.usable()` drops it, `ResultStore.missing()` puts its seed back in
the queue.

What the fingerprint cannot see is worth knowing: dynamic imports and Hydra `_target_` strings,
dispatch by attribute, and everything that is not Python — config YAML, downloaded datasets, and the
versions of torch and numpy underneath.

---

## Getting from a claim to the campaigns behind it

Every public claim should be traceable to specific stored records, and the trace is derivable from
the code rather than from anyone's notes.

1. **Find the tier.** Only `Purpose.BENCHMARK` tiers carry claims. Everything else is a diagnostic
   that explains behaviour, or a selection tier that fixed our own configuration.
2. **Read the task's `purpose`.** It is a required field and it says what that task decides that the
   others cannot — which is the claim the task is entitled to support.
3. **Open the files.** `results/<task>/<method>.jsonl`, one record per seed.

The whole map prints from the library:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'experiments')
from run_suite import tiers
for tier in tiers(100, 50):
    print(f'{tier.name}  [{tier.purpose}]  carries claims: {tier.headline}')
    for task in tier.tasks:
        print(f'    results/{task.name}/  --  {task.purpose}')
"
```

Two things to keep straight while reading the output. The selection phase stores its campaigns under
the same task name as the objectives diagnostic (`objectives`) and is distinguished by arm name, so
the arm is what tells you whether a record chose our configuration or measured a method. And an arm
sitting on a task's audited optimum is marked `SOLVED` in the report: a task an arm has already
exhausted cannot rank the arms above it, and a comparison drawn on it is labelled vacuous rather than
printed as a p-value.

---

## What is implemented, and what is in flux

**Implemented and exercised by tests:** fitness landscapes (Ehrlich, GB1, TrpB, CH65, multi-objective
Ehrlich) with noise/budget/cache wrappers; a mutation environment with feasibility masking;
GFlowNet training objectives (trajectory balance, contrastive balance, detailed balance,
sub-trajectory balance, forward-looking DB, Genetic-GFN); classical baselines (random, hill climbing,
traditional directed evolution as `single-step` and `recomb`, MLDE, ALDE, AdaLead, genetic,
simulated annealing, CMA-ES, NSGA-II); deep-ensemble surrogate and acquisition rules;
the design–build–test–learn campaign loop with re-anchoring; multi-objective rewards and Pareto
metrics; the benchmark harness, result store and selection phase; a Hydra CLI (`evogfn train`,
`evogfn campaign`).

**In flux, and not to be relied on:**

* **The arm list.** The suite is mid-transition from "every classical baseline silently receives a
  surrogate" to "every baseline is the pipeline its paper describes, with one named ablation ladder"
  (`genetic`, `genetic+screen`, `genetic+search`, `genetic+distinct`, `random+screen`). Read the arms
  out of `evogfn.benchmark.methods` rather than out of any document.
* **The stored results.** Re-anchoring, the attainable-optimum audit and the arm-list change all
  landed recently. Most of the store is stale against them and has not been re-run.
* **The selected configuration.** None is currently recorded; see above.
* **The public API.** Not stable.

[`docs/limitations.md`](docs/limitations.md) is the ledger: retracted claims, void measurements,
controls that were never run, and defects in the machinery that produces the numbers. Read it before
quoting anything.

---

## Design

Every major piece is an interface with swappable implementations, so adding a landscape, a sampler or
an acquisition function means implementing one class and pointing a config at it.

| Component | Replace it to... |
|---|---|
| `FitnessLandscape` | score sequences with your own assay data, model, or simulator |
| `SequenceEnvironment` | change how variants are built — the shipped one mutates a parent under a budget |
| `Sampler` | swap the search method — GFlowNets and baselines share one interface |
| `Acquisition` | change how a batch is chosen under uncertainty |
| `Tracker` | send metrics somewhere other than the console |

Landscapes compose: measurement noise, an evaluation budget and caching are wrappers you apply to any
of them, so a budget cannot be accidentally bypassed.

---

## Documentation

- [Getting started](docs/getting-started.md) — install, run a campaign, train a policy, bring your own landscape.
- [The benchmark suite](docs/benchmark.md) — what each task decides, what a protocol is, how to run it.
- [Choosing the configuration](docs/selection.md) — the selection rule, and what a staged design cannot see.
- [What this does not show](docs/limitations.md) — the ledger.

---

## License

Copyright © 2026 Simon J. Crouzet. Licensed under the **Apache License 2.0**.

You may freely use, modify, and distribute this software — including for commercial purposes —
provided that you preserve the copyright notice and license text in any distribution. See
[`LICENSE`](LICENSE) for the full terms.

---

## About

I'm Simon Crouzet, an independent researcher and consultant in AI/ML for molecular design and drug
discovery. EvoGFN came out of a long-standing interest in directed evolution, and in GFlowNets —
and in what happens when you stop treating the first as an optimisation problem and start treating it
as the sampling problem the second was built for.

If you find this useful, have ideas, or are working on something in the same space and want to
exchange — feel free to reach out. I'm also available for project-based work in computational molecular
design and ML workflow development.

- **GitHub:** [@simoncrouzet](https://github.com/simoncrouzet)

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue to discuss
significant changes before submitting a pull request. All pull requests should include tests and pass
the existing suite. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup and conventions.

---

## Credit & Citation

EvoGFN is open source under the Apache 2.0 License. You are free to use it in research and
commercial work — please credit the original project and respect the license terms.

If you use EvoGFN in your work, please acknowledge it and feel free to get in touch.

---

## References

- Bengio, E., Jain, M., Korablyov, M., Precup, D. & Bengio, Y. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. *NeurIPS* 34, 27381–27394.
- Malkin, N., Jain, M., Bengio, E., Sun, C. & Bengio, Y. (2022). Trajectory Balance: Improved Credit Assignment in GFlowNets. *NeurIPS* 35.
- Jain, M. et al. (2022). Biological Sequence Design with GFlowNets. *ICML*.
- Jain, M. et al. (2023). Multi-Objective GFlowNets. *ICML*.
- Stanton, S., Alberstein, R., Frey, N., Watkins, A. & Cho, K. (2024). Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms. *ICML Workshop on ML for Life and Material Sciences*.
- Wu, N.C., Dai, L., Olson, C.A., Lloyd-Smith, J.O. & Sun, R. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife* 5, e16965.
- Yang, K.K., Wu, Z. & Arnold, F.H. (2019). Machine-learning-guided directed evolution for protein engineering. *Nature Methods* 16, 687–694.
</content>
