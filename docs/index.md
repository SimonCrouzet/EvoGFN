# EvoGFN

**EvoGFN** is a Python library for in-silico directed evolution with Generative Flow
Networks (GFlowNets). It generates batches of sequence variants that are diverse and
high-fitness at the same time, rather than many near-copies of a single best hit.

Use it to run directed-evolution campaigns against a fitness landscape — your own, or one of
the built-in benchmarks — and to compare a GFlowNet against classical baselines (traditional
directed evolution, MLDE, ALDE, AdaLead, a genetic algorithm, hill climbing, CMA-ES) on equal
terms.

!!! warning "Early development, and no results on this site"
    The API is not stable, and the benchmark suite is mid-change: the search radius, the
    anchoring rule and the arm list have all moved recently, which invalidates most of what is
    stored. These pages therefore describe **design, protocol and rationale only** — no results.
    The numbers live in `results/` and are regenerated into tables once the pipeline settles.
    The one page that still quotes figures is [what this does not show](limitations.md), which
    states each one's status alongside it. It is not an appendix — read it before quoting
    anything.

---

## The argument

A design round has a fixed budget. You can synthesise and assay so many variants and no more.
If your proposal method returns 96 sequences that are minor variations on one design, you have
spent the budget on a single bet, and anything that kills that design kills the whole round.

Optimisers do this by construction: they climb toward one peak. A GFlowNet learns a policy
that samples variants *in proportion to* their predicted fitness, so a batch spreads across
the high-fitness regions of the landscape instead of piling onto one. Applied to mutation
trajectories from a parent sequence, that is a direct model of what directed evolution
actually does.

There is a second argument, and structurally it is the sharper one. Many sequence design
problems have a **constructibility constraint**: most strings are not things you could build.
A method that proposes designs and filters them afterwards spends its budget on wells that
return nothing. A GFlowNet built on a masked construction graph cannot generate an infeasible
design at all — feasibility is a property of the graph, not a post-hoc check.

That argument is structural rather than measured. Masking is feasible *by construction*, which
is definitional and not an achievement; whether it also **searches better** than rejection
sampling at a matched budget is an empirical question, and it is the one the suite exists to
answer.

---

## What this site does and does not contain

Stated up front, because the rest of the documentation is easier to read honestly if the
headline is honest first.

**The design pages carry no measured figures**, and that is a decision rather than an
omission. A number copied into prose has nothing tying it to the run that produced it, and
every false claim a previous audit found on this site was exactly that: a hand-copied figure
that outlived its experiment. Design descriptions held up; measurements did not. So the prose
here says what each test is *for* and why it is built that way, and the numbers stay in
`results/`, where every record carries a fingerprint of the code that produced it and is
dropped from any table once that code moves.

The exception is [what this does not show](limitations.md), which is the ledger: retracted
claims, void measurements, controls that were never run, and defects in the machinery. Where
it quotes a figure it states that figure's standing in the same sentence. Read it before
quoting anything from the other pages.

---

## Where to start

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install, run a campaign, train a policy.
- **[The benchmark suite](benchmark.md)** — what each task decides, and what a protocol is.
- **[Choosing the configuration](selection.md)** — how the GFlowNet's settings are fixed, before anything is measured.
- **[What this does not show](limitations.md)** — the limits, in one place.
- **[API reference](reference/landscapes.md)** — generated from the source.

</div>

!!! note "One gap in the API reference"
    The reference pages are generated from the docstrings in `src/`. Module-level constants
    — `MAX_ENUMERABLE_SIZE`, `MIN_BINS`, `WET_LAB_PROTOCOLS`, the `PUBLISHED_*` values in
    `baselines.mlde` and the rest — are documented with `#:` comments, which the Markdown
    docstring parser does not pick up, so they do not appear on these pages and references to
    them read as plain names rather than links. Their values and the reasoning behind them are
    in the source.

---

## Design

Every major piece is an interface with swappable implementations, so adding a landscape, a
sampler or an acquisition function means implementing one class and pointing a config at it.

| Component | Replace it to... |
|---|---|
| [`FitnessLandscape`](reference/landscapes.md) | score sequences with your own assay data, model, or simulator |
| [`SequenceEnvironment`](reference/environments.md) | change how variants are built — the shipped one mutates a parent under a budget |
| [`Sampler`](reference/algorithms.md) | swap the search method — GFlowNets and baselines share one interface |
| [`Acquisition`](reference/support.md) | change how a batch is chosen under uncertainty |
| [`Tracker`](reference/support.md) | send metrics somewhere other than the console |

Landscapes compose: measurement noise, an evaluation budget and caching are wrappers you
apply to any of them, so a budget cannot be accidentally bypassed.

---

## Licence and citation

Copyright © 2026 Simon J. Crouzet. Licensed under the **Apache License 2.0**. You may use,
modify and distribute this software including for commercial purposes, provided you preserve
the copyright notice and licence text.

If you use EvoGFN in your work, please credit the project and feel free to get in touch —
[@simoncrouzet](https://github.com/simoncrouzet).

---

## References

- Bengio, E., Jain, M., Korablyov, M., Precup, D. & Bengio, Y. (2021). Flow Network based
  Generative Models for Non-Iterative Diverse Candidate Generation. *NeurIPS* 34, 27381–27394.
- Malkin, N., Jain, M., Bengio, E., Sun, C. & Bengio, Y. (2022). Trajectory Balance: Improved
  Credit Assignment in GFlowNets. *NeurIPS* 35.
- Jain, M. et al. (2022). Biological Sequence Design with GFlowNets. *ICML*.
- Jain, M. et al. (2023). Multi-Objective GFlowNets. *ICML*.
- Stanton, S., Alberstein, R., Frey, N., Watkins, A. & Cho, K. (2024). Closed-Form Test
  Functions for Biophysical Sequence Optimization Algorithms. *ICML Workshop on ML for Life
  and Material Sciences*.
- Wu, N.C., Dai, L., Olson, C.A., Lloyd-Smith, J.O. & Sun, R. (2016). Adaptation in protein
  fitness landscapes is facilitated by indirect paths. *eLife* 5, e16965.
- Yang, K.K., Wu, Z. & Arnold, F.H. (2019). Machine-learning-guided directed evolution for
  protein engineering. *Nature Methods* 16, 687–694.
