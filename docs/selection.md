# Choosing the GFlowNet's configuration

Every classical baseline in this suite runs at hyperparameters its own authors tuned. The
genetic algorithm uses the mutation and recombination rates from the Ehrlich paper; MLDE runs
in the regime Wittmann et al. actually run it in. The GFlowNet had no such authority to appeal
to: gradient steps set to whatever was fast enough, a reward exponent carried over from a
different paper on a different problem, and a training objective that was never chosen at all.

A table built that way reports a comparison of *configurations* while claiming to compare
*methods* — and it does it in the direction that flatters the field, because the field is tuned
and we are not.

So the configuration is **selected**, by a rule written down before any of its numbers existed,
on a landscape no claim is ever drawn from. This page is that procedure: the rule, where it is
allowed to run, what each stage decides, what the design cannot see, and how to reproduce it.

---

## The rule

**Lowest mean regret, with top-K diversity breaking statistical ties.**

Both halves were fixed before the first campaign ran. A criterion chosen after seeing the table
is not a criterion, and "best regret, except on the axis where diversity looked better" is how a
sweep becomes a story.

Two things about the rule need to be precise, because loose versions of both are common.

**A tie is statistical, not numerical.** Two arms tie when a paired comparison over the seeds
they share cannot separate them. That is a statement about the evidence, not about decimal
places: an arm ahead by 0.006 with a confidence interval spanning zero has not won anything.
Arms with a *worse* mean stay in the running when the gap is inside the noise, because losing by
less than the noise is not losing.

**The tie-break is load-bearing, not tidying.** What this project claims is diverse, feasible,
high-fitness variants — not high-fitness ones. A rule reading regret alone would happily select
a configuration that optimises well and samples badly, and the diversity column of the headline
table would then have to live with whatever that produced. Ties are not a rare edge case here:
at 30 seeds on the diagnostic landscape, four of five objectives sat within 0.02 regret of each
other.

Two smaller decisions are part of the rule rather than implementation detail:

* Only seeds that **every** arm holds are used, so the paired comparisons deciding ties are
  genuinely paired.
* An arm that failed on some seeds is scored on the ones it survived; an arm that failed on all
  of them is not eligible to be chosen.

---

## Where it runs, and where it must not

Both the objective comparison and the scans run on the **diagnostic landscape** — Ehrlich at
`L = 32`, four rounds of 96, the same cheap instance the other diagnostics vary an axis on. No
headline task uses it.

That is the whole reason the phase is not tuning on the test set. The configuration is fixed
before a single claim-carrying campaign is scored, and it is fixed against a landscape that
carries no claim. The tier is tagged `Purpose.SELECTION` rather than `DIAGNOSTIC`, which keeps
the distinction in the type: a diagnostic measures how methods behave, a selection tier chooses
*our own* configuration, and the results table can refuse a tier that was never eligible to
appear in it.

---

## Three stages

| Stage | Decides | Arms | Seeds |
|---|---|---|---|
| **A** | the training objective | six candidates, at the default reward exponent | 100 |
| **B** | the reward exponent, for whichever objective A chose | `SELECTION_BETAS` | 100 |
| **C — screen** | *nothing.* Nominates candidates | 100 random points of each screened objective's joint space | 10 |
| **C — confirm** | the joint configuration | the finalists, **plus the incumbent** | 100 |

**Why 100 seeds.** The 30-seed diagnostic put four objectives within 0.02 regret of each other
and asked for thousands of seeds to separate the closest pairs. One hundred is what it said
would resolve the single gap that looked real — sub-trajectory balance against trajectory
balance — and it is enough to state honestly that the rest are tied.

**Why A and B are sequential.** The full cross of objectives and exponents is far more compute
than the question needs, and B cannot start before its input exists: the winning objective is not
known until A has finished.

**Why C is not.** Moving one axis at a time cannot see interactions, and — worse — it never
reaches the parameters that make the leading objectives *families* rather than points.
Sub-trajectory balance's `lam` interpolates detailed balance at one end and trajectory balance at
the other; Genetic-GFN's `mix` degenerates to an ordinary GFlowNet at one end. Selecting either
objective without moving its own defining parameter is the same fault as comparing objectives at
another objective's inherited defaults, which is what this whole phase exists to avoid.

### Stage B's grid was widened once, deliberately

The first exponent pass came back monotone to its own edge — 0.502, 0.473, 0.446 across
`beta` 1, 3, 10 — which cannot distinguish "10 is right" from "10 is the largest value we
offered". Widening upward answered it: regret turns hard above 10 and the optimum is interior.
The values below 1 close the same hole at the other end, where diversity was still rising at the
lowest exponent tried, and diversity is the axis the tie-break actually decides on.

Extending a grid after looking at it is only legitimate under conditions that happen to hold
here: the rule was fixed before any of these numbers existed, the landscape carries no claim,
and the rule is regret-first — an exponent buying diversity at a real regret cost is not
eligible, since only *statistical* ties go to diversity. Extending until the answer becomes
agreeable would not be legitimate, so the grids are fixed now and the full curve is reported
either way.

---

## Stage C: screen wide and cheap, confirm narrow and rigorous

### The space

Two objectives get a screen, not one. Stage A compared all six candidates at a single shared set
of inherited defaults, which was symmetric and therefore fair. Screening only the winner would
put a tuned arm against untuned ones and reintroduce exactly the asymmetry this phase exists to
remove. Genetic-GFN is where that bites hardest: its own defining parameter has never been moved,
and its premise — that directed evolution *is* a genetic algorithm — is the strongest prior
reason any variant here had to work. "Tuned on its own axes and still behind" is a far stronger
statement than one drawn from a single inherited configuration.

| Axis | Values | Applies to |
|---|---|---|
| `steps` | 50, 100, 150, 200, 300, 450 | both |
| `hidden_dim` | 64, 128, 256 | both |
| `beta` | 0.1, 0.2, 0.3, 0.4, 0.5, 1 | both |
| `lam` | 0.1, 0.2, 0.5, 0.75, 0.9 | `gfn-subtb` |
| `mix` | 0.1, 0.25, 0.5, 0.75, 0.9 | `genetic-gfn` |

540 points each, sampled at 100 — a fifth of the space, which is a screen rather than a grid with
extra words. Random rather than grid because the space has few effective dimensions and
conditional structure: a grid spends its budget re-measuring the axes that do not matter, while
every random draw is a fresh value on every axis at once. The draw is a function of
`SCREEN_SAMPLING_SEED` and the grids alone, so the candidate set can be redrawn and checked.

Three choices in that table are worth stating rather than leaving to be inferred.

`beta` is here **despite** stage B, because stage B measured it at one step count. If the screen
prefers a much smaller step count the exponent need not still be right, and that is the one
interaction there is prior reason to expect. It is deliberately narrower than `SELECTION_BETAS`
and narrow *downward*: above the flat region stage B's curve degrades monotonically, on far more
seeds per point than a screen can afford, so spending screen configurations there would re-ask a
settled question worse. The top of the plausible range is kept as a control so the screen can
still show an optimum shifting upward.

`mix` excludes both endpoints, and neither exclusion is taste. At zero the teacher contributes
nothing and the arm *is* plain trajectory balance carrying unused machinery; at one the policy
sees only bred offspring and never its own samples.

`steps` reaches one point above the shipped setting. Gradient steps × batch size is the number of
*proxy* evaluations the GFlowNet spends per round, and proxy spend is a reported column — so this
is a number the paper prints, and an optimum pinned at the edge of what was offered would not be
an answer.

### The screen nominates. It never ranks.

**No number the screen produces is reported anywhere.** Ten seeds on a coarse discrete objective
is not evidence that can order anything, and the screen's output is a *set of candidates*, not a
ranking of them. Each screen prints how many of its configurations a paired comparison cannot
separate from its own leader, so the size of the plausible set is visible before the confirmation
is paid for rather than after.

The finalists are named on the command line. Allocating a fixed confirmation budget between two
objectives is a judgement about how much a ten-seed ordering is worth, not an arithmetic, so the
script prints a proposal and a person decides. The proposal is two-valued — an even split where a
paired comparison cannot separate the screens' leaders, and otherwise the weaker screen keeps a
floor of 2 while the rest goes to the better one. The floor exists because the comparison being
made is between two *winners of a search*, which is biased toward whichever screen sampled
luckier; and because a ten-seed screen is not evidence enough to write an objective off entirely.

### The incumbent is always confirmed

Whatever the screen thought of it, the standing configuration is measured alongside the finalists
at the full seed count, through the same pre-declared rule. That is the whole safety argument: the
screen can only ever **add** a better configuration, never displace the current one on evidence
too thin to displace anything.

---

## What this design cannot see

Stages A and B are sequential, so they cannot see interactions between the objective and its
hyperparameters. An objective that loses at the default exponent and would have won at another
one is invisible to them.

That is not a hypothetical worry. The reward-exponent curve **reversed direction** between
trajectory balance and sub-trajectory balance, so these hyperparameters demonstrably do not
transfer across the objective.

Stage C closes part of that hole and does not pretend to close all of it:

* It moves every axis at once, for two objectives rather than one, so an interaction between them
  is at least reachable.
* It screens two objectives, not six. Re-opening stage A's choice on ten noisy seeds would trade
  a 100-seed measurement for a 10-seed one, so the other four are still known at inherited
  defaults only.
* A fifth of each space is sampled, so a good configuration can simply not have been drawn. The
  screen is a lower bound on what the space contains, never an upper one.

The design's remaining mitigations are partial, and worth naming as partial:

* Stage A fixes the exponent at the default the objectives were compared at, rather than at an
  arbitrary value, so the comparison is at least at a shared and stated point.
* Every stage reports its whole table, not just its winner, so a reader can see how wide the
  ties were.

---

## Running it

```bash
uv run python experiments/select_configuration.py            # every stage
uv run python experiments/select_configuration.py --report   # read the store, run nothing
```

Each stage prints its full table — regret and diversity per arm, which arms tied, which was
chosen, and the sentence saying why — before moving to the next.

| Flag | What it does |
|---|---|
| `--report` | Read what is already stored and print the tables. Runs no campaigns |
| `--stage {a,b,c,c-screen,c-confirm,both}` | Run one stage. Shards use `a`, then one process runs `b`; stage C splits into its screen and its confirmation |
| `--only ARM` | Run just this arm. Repeatable; this is the sharding knob |
| `--print-winner` | Print stage A's chosen arm and exit, for a coordinator to shard stage B on |
| `--seeds N` | How many seeds an arm must hold before a stage is allowed to choose |
| `--screen-size N` | Configurations drawn per screened objective |
| `--screen-seeds N` | Seeds per screened configuration |
| `--screen-seed N` | Seed for the draw, so the candidate set can be reproduced |
| `--confirm ARM` | A finalist. Repeatable, and **required** before the confirmation runs |
| `--confirm-slots N`, `--confirm-floor N` | Shape the printed allocation proposal |
| `--results DIR` | Where results live. Defaults to `results/` |

Stage C stops after the screen and says so. It is waiting on `--confirm`, because the finalists
are a person's call and not the script's.

The phase refuses to start if threading is not pinned (exit code 3), because an unpinned run
produces numbers a later run cannot reproduce. A stage that cannot yet choose — because some arm
is short of seeds — says so and stops rather than choosing on partial evidence.

### It is resumable

Every campaign is written the moment it finishes, so an interrupted run keeps everything up to
the interruption and a rerun computes only what is missing. Records also carry a fingerprint of
the code that could have produced them, so a stored campaign that the source has since moved
past is re-run rather than trusted.

### It shards, one process per arm

```bash
# Stage A: one process per objective, in parallel.
for arm in gfn-tb gfn-contrastive genetic-gfn gfn-db gfn-subtb gfn-fldb; do
  uv run python experiments/select_configuration.py --stage a --only "$arm" &
done
wait

# Whichever objective won stage A is what stage B scans.
winner=$(uv run python experiments/select_configuration.py --print-winner)

# Stage C's screen has far more arms than cores, so it shards by seed slice
# instead: every process runs every arm over its own share of the ten seeds.
for lo in 0 2 4 6 8; do
  uv run python experiments/select_configuration.py \
    --stage c-screen --seed-from "$lo" --seed-to "$((lo + 2))" &
done
wait
```

Sharding across *processes* rather than threads is what makes this safe. Campaigns are
independent, the store keeps one file per arm so writers never collide, and every campaign is
seeded from its own seed rather than from process order — so a sharded run and a serial one
produce identical records. Raising the thread count instead would not, since a multithreaded
reduction sums in completion order.

A shard that finishes first stops after its own arms: a winner drawn from a subset is a winner
of the subset, not of the stage.

---

## What comes out

The phase writes `results/selected.json`:

| Field | Meaning |
|---|---|
| `objective`, `beta`, `steps`, `lam`, `mix`, `hidden_dim` | the chosen configuration, as fields rather than a name to be parsed. `null` means this objective reads no such knob |
| `arm` | the arm's name, as it appears in the results table |
| `objective_reason`, `arm_reason`, `confirmed_reason` | why each stage chose what it chose, in a form that can be pasted into a caption |
| `objective_tied` | the arms stage A could not separate. More than one name means the choice was settled on diversity rather than on regret |
| `screen_seed`, `screen_size`, `screen_seeds`, `finalists` | how the candidate set was drawn and which of it went forward, so the nomination can be redrawn. The screen's own numbers are deliberately absent: it nominated, and nothing it measured is reported |
| `seeds`, `task` | what the decision rests on |

Arm names for stage C encode the whole configuration — `gfn-subtb@b0.3-s300-l0.9-h128` — because
they are store keys, and a key that changes orphans every record written under the old one. They
are for *identity only*: the settings are read back from the record's own `parameters` field,
which is what the campaign actually ran at, and the name is parsed alongside purely so a
disagreement between the two raises instead of passing silently.

The benchmark **reads** that file rather than re-deriving the choice. Re-deriving would silently
pick a different arm the moment a seed count or an arm list moved, and the table would then
report a configuration no selection ever made. The selected arm *replaces* the untuned GFlowNet
arms in the headline tiers rather than joining them — keeping both would put two configurations
of the same method in one table, and the better of the two would be exactly the thing the phase
was run to avoid reporting.

If the file is absent, the suite falls back to the untuned defaults and says so, rather than
quietly benchmarking them as though they had been chosen.

---

## API

- [`evogfn.benchmark.selection`](reference/benchmark.md) — the rule and the arm builders.
- [The benchmark suite](benchmark.md) — what each task decides, and what a protocol is.
- [What this does not show](limitations.md) — including what this staged design cannot see.
