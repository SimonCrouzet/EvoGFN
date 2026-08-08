"""What a campaign costs, in the units a wet lab actually works in.

A protocol is three numbers -- rounds, batch size, mutation budget -- and the
reason this is a class rather than three loose arguments is that their *product*
is the only quantity any claim can be indexed by. A result reported without its
budget cannot be compared to one that has it, and the surveyed literature shows
the field routinely comparing across budgets that differ by two orders of
magnitude.

The grounding
-------------

======================================  ========  ======  ======
Campaign                                per round  rounds   total
======================================  ========  ======  ======
EVOLVEpro (*Science* 2025)                 11-12     4-8   50-90
LaMBO-2 wet lab                             ~125       3     374
ALDE (Arnold lab, *Nat Commun* 2025)    216/90/90       3     396
MLDE / ftMLDE (Wittmann 2021)             384+96       2     480
CLADE (Qiu & Wei 2021)                        96       5     480
TrpB (Buller, *PNAS* 2015), classical   528/1408/1144   3   ~3,080
======================================  ========  ======  ======

Against which the machine-learning convention -- 10 rounds of 100 or 128, 15 of
256 on the harder GFP splits, and 10,000 for GFN-AL on AMP -- sits above even
*classical* directed evolution. The sharp version: MLDE's entire claim is
reaching the answer in ~480 assays instead of ~3,000, and a benchmark run at
10,000 has given that back before the first comparison is made.

And these are the *charitable* figures. `ML_CONVENTION` counts what the active
rounds spend and nothing else, while several of the methods behind those rows
are handed a labelled offline corpus before round one -- 1,024 to 32,898
sequences for delta-CS, depending on the task -- which no budget any of them
reports includes.

`WET_LAB_PROTOCOLS` names the real ones so an experiment can cite a
campaign rather than a round number someone liked.
"""

from __future__ import annotations

from dataclasses import dataclass

#: One 96-well plate: the unit a screening campaign is actually quantised into.
PLATE = 96

#: Rounds are bounded by turnaround time, not by budget. Published campaigns run
#: 2-8; beyond that the calendar, not the assay, is the constraint.
MAX_SENSIBLE_ROUNDS = 8


@dataclass(frozen=True, slots=True)
class Protocol:
    """The experimental design a comparison is run under.

    Attributes:
        rounds: Design-build-test-learn cycles.
        batch_size: Variants measured per round.
        max_mutations: How far from the parent a design may stray. Below the
            sequence length this is a real constraint; at or above it every
            sequence is reachable and the mutation budget does nothing, which
            is worth knowing before reading a result as evidence about
            constrained search.
        label: Name of the campaign this imitates, when it imitates one.

    Raises:
        ValueError: If any size is not positive.
    """

    rounds: int
    batch_size: int
    max_mutations: int | None = None
    label: str = ""

    def __post_init__(self) -> None:
        """Reject designs that cannot describe an experiment."""
        if self.rounds < 1:
            raise ValueError(f"rounds must be at least 1, got {self.rounds}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.max_mutations is not None and self.max_mutations < 1:
            raise ValueError(f"max_mutations must be at least 1, got {self.max_mutations}")

    @property
    def budget(self) -> int:
        """Total oracle calls: the number every claim is indexed by."""
        return self.rounds * self.batch_size

    @property
    def plates(self) -> float:
        """Budget expressed in 96-well plates, which is what gets ordered."""
        return self.budget / PLATE

    def constrains_search(self, sequence_length: int) -> bool:
        """Whether the mutation budget actually restricts the reachable set.

        Args:
            sequence_length: Length of the sequences being designed.

        Returns:
            ``False`` when every sequence is reachable, in which case a result
            says nothing about search under a mutation constraint. On GB1 --
            four sites, four mutations -- this is ``False``, so that landscape
            exercises the easiest possible geometry.
        """
        return self.max_mutations is not None and self.max_mutations < sequence_length

    def __repr__(self) -> str:
        """Name the protocol by its shape and total."""
        name = f"{self.label}: " if self.label else ""
        return f"{name}{self.rounds}x{self.batch_size}={self.budget}"


#: Protocols taken from real campaigns, so an experiment can cite one.
WET_LAB_PROTOCOLS: tuple[Protocol, ...] = (
    Protocol(rounds=8, batch_size=12, label="EVOLVEpro"),
    Protocol(rounds=3, batch_size=125, label="LaMBO-2"),
    Protocol(rounds=3, batch_size=132, label="ALDE"),
    Protocol(rounds=5, batch_size=PLATE, label="CLADE"),
    Protocol(rounds=4, batch_size=PLATE, label="four plates"),
)

#: The machine-learning convention, included so the gap to a wet-lab budget can
#: be measured rather than asserted.
#:
#: Every figure here is **oracle queries spent by the active rounds, and nothing
#: else**. That is the narrow reading and it is the one that flatters these
#: papers, because several of them are handed a labelled corpus before round one
#: that none of these numbers count. delta-CS is the sharpest case: it takes an
#: initial offline dataset as an argument to its algorithm, and across the tasks
#: it reports that dataset runs from 1,024 to 32,898 labelled sequences --
#: between two and eighty-six times this suite's entire 384-assay budget, spent
#: before its first query. The same holds for CbAS, DbAS and MINs, which is why
#: none of them is run here. A budget column that showed only the round spend
#: would therefore *understate* the gap it exists to measure, and the paragraph
#: in ``docs/limitations.md`` is where that is stated rather than implied.
ML_CONVENTION: tuple[Protocol, ...] = (
    Protocol(rounds=10, batch_size=100, label="AdaLead/PEX/DyNA-PPO"),
    # delta-CS runs *two* published protocols, not one, and an entry carrying
    # only the first would let a comparison be drawn against whichever number
    # suited it. This is the RNA / TF-Bind-8 / GFP / AAV setting.
    Protocol(rounds=10, batch_size=128, label="delta-CS/SILO"),
    # The second: Kirjner et al.'s GFP-medium and GFP-hard, where delta-CS
    # follows LatProtRL's protocol instead of its own. Three times the budget of
    # the row above it, on the benchmark whose difficulty is the point -- so a
    # result quoted from those tasks was bought at 3,840 queries and must not be
    # read against the 1,280 the other row names.
    Protocol(rounds=15, batch_size=256, label="delta-CS (GFP-medium/hard, after LatProtRL)"),
    Protocol(rounds=10, batch_size=1000, label="GFN-AL (AMP)"),
)


def round_sweep(budget: int, *, batch_size: int = PLATE) -> tuple[Protocol, ...]:
    """Protocols that spend the same budget over different numbers of rounds.

    The one experiment nobody has run at a three-digit budget: the published
    work that varies round shape at a fixed budget disagrees about whether it
    matters at all. The disagreement tracks how data-rich the first round is,
    which predicts the effect is largest exactly where the wet lab operates.

    Args:
        budget: Total oracle calls to hold fixed.
        batch_size: Largest batch to consider, halved down from there.

    Returns:
        Protocols from many small rounds to one large one, each spending as
        close to ``budget`` as an integer number of rounds allows.

    Raises:
        ValueError: If the budget or batch size is not positive.
    """
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    protocols = []
    size = batch_size
    while size >= 1:
        rounds = max(1, budget // size)
        if rounds <= MAX_SENSIBLE_ROUNDS:
            protocols.append(Protocol(rounds=rounds, batch_size=size, label=f"{rounds}x{size}"))
        size //= 2
    return tuple(reversed(protocols))
