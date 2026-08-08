r"""holo-bench's own Ehrlich instances, made scoreable by this package.

Why this module exists
----------------------

The manuscript records, as a limitation, that our Ehrlich instances are not
drawn from the same distribution as Stanton et al.'s reference implementation:
they chunk one Markov draw of length $c \cdot k$ into motifs and space them by a
simplex draw, we carve motifs out of a full-length feasible sequence in disjoint
blocks with uniform gaps. That is a real difference and no amount of reading the
paper settles it. The only thing that settles it is running both.

Running both is awkward for one reason: ``pytorch-holo`` pins ``numpy<2.0``, so
it cannot be installed beside this package. It lives in its own environment and
is reached by subprocess -- see
[_holo_worker][evogfn.landscapes._holo_worker] for the wire format. The
interpreter is found from ``$EVOGFN_HOLO_PYTHON`` or from
`DEFAULT_INTERPRETERS`, and everything here raises
[HoloUnavailableError][evogfn.landscapes.holo_reference.HoloUnavailableError] rather than
falling back to a hand-rolled substitute. A "reference" we wrote ourselves would
answer the question by assuming it.

What crossing the boundary buys
-------------------------------

[HoloEhrlichLandscape][evogfn.landscapes.holo_reference.HoloEhrlichLandscape]
is an [EhrlichLandscape][evogfn.landscapes.ehrlich.EhrlichLandscape] whose
motifs, spacings and transition matrix are the reference instance's rather than
ours. That subclassing is load-bearing rather than cosmetic: the attainability
audit and the mutation environment both dispatch on
``isinstance(landscape, EhrlichLandscape)``, so a reference instance wrapped
this way runs through the *unmodified* audit, and every number it produces is
the audit's, not a reimplementation's.

The construction bypasses ``EhrlichLandscape.__init__`` because that constructor
*generates* an instance and there is no seam to inject one. The attributes it
would have set are set here instead, and
[HoloEhrlichLandscape.installed_attributes][evogfn.landscapes.holo_reference.HoloEhrlichLandscape.installed_attributes]
names them so a test can assert the two agree -- if ``ehrlich.py`` grows or
renames a field, that test fails rather than this adapter silently scoring
against a stale one.

Two conversions happen at the boundary, and both are places an off-by-one would
be invisible:

* holo stores **gaps** between consecutive motif elements (length $k-1$); we
  store **cumulative offsets** from the placement (length $k$, leading zero).
* holo's quantisation is ``count // ceil(k/q) / (k / ceil(k/q))``, ours is
  ``count // (k // q) / q``. These coincide exactly when $q$ divides $k$ and
  differ otherwise, so this module refuses instances where it does not rather
  than reporting a comparison whose two sides use different arithmetic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from evogfn.core.types import Alphabet
from evogfn.landscapes.ehrlich import EhrlichLandscape

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Tokens

#: Environment variable naming a Python that can ``import holo``.
INTERPRETER_ENV_VAR = "EVOGFN_HOLO_PYTHON"

#: Where a holo environment is looked for when the variable is unset. These are
#: paths, not a package name, precisely because holo cannot be installed into
#: this project's environment; ``sys.executable`` is included last for the case
#: where someone has managed it anyway.
DEFAULT_INTERPRETERS: tuple[str, ...] = (
    "/mnt/wsl/ext/envs/holo/bin/python",
    str(Path.home() / ".venvs" / "holo" / "bin" / "python"),
    "/opt/holo/bin/python",
)

#: Draws of the reference's own ``initial_solution`` kept per instance, so a
#: campaign anchored on this landscape starts from a wild type holo drew rather
#: than one we drew from holo's chain.
INITIAL_SOLUTION_POOL = 16

_WORKER = Path(__file__).with_name("_holo_worker.py")


class HoloUnavailableError(RuntimeError):
    """No interpreter on this machine can import ``holo``.

    Raised rather than degrading to an approximation: the entire value of this
    module is that the other side of the comparison is code we did not write.
    """


def find_interpreter() -> Path | None:
    """Locate a Python that can import ``holo``.

    Returns:
        The interpreter's path, or ``None`` if none of the candidates works.
        Candidates are probed by actually importing ``holo`` in them, because a
        virtualenv that merely exists is the common failure here.
    """
    candidates = [os.environ[INTERPRETER_ENV_VAR]] if INTERPRETER_ENV_VAR in os.environ else []
    candidates.extend(DEFAULT_INTERPRETERS)
    candidates.append(sys.executable)
    for candidate in candidates:
        resolved = shutil.which(candidate) or candidate
        if not Path(resolved).exists():
            continue
        probe = subprocess.run(  # noqa: S603 - the command is a probed interpreter path
            [resolved, "-c", "import holo.test_functions.closed_form"],
            capture_output=True,
            check=False,
            timeout=120,
        )
        if probe.returncode == 0:
            return Path(resolved)
    return None


def require_interpreter() -> Path:
    """Locate the holo interpreter or explain what to install.

    Returns:
        The interpreter's path.

    Raises:
        HoloUnavailableError: If no candidate can import ``holo``.
    """
    interpreter = find_interpreter()
    if interpreter is None:
        raise HoloUnavailableError(
            "no interpreter found that can import holo. pytorch-holo pins numpy<2.0 and so "
            f"cannot share this project's environment; create one for it and point "
            f"${INTERPRETER_ENV_VAR} at its python, or place it at one of "
            f"{', '.join(DEFAULT_INTERPRETERS)}"
        )
    return interpreter


@dataclass(frozen=True)
class ReferenceParameters:
    """The arguments holo's ``Ehrlich`` is constructed from.

    Named separately from the instance because they have to survive a JSON
    round trip to another interpreter, and because scoring a batch through the
    reference means rebuilding it there from exactly these.

    Attributes:
        num_states: Alphabet size. holo derives the transition matrix's
            bandwidth from this as ``int(0.4 * num_states)``, so it is also the
            only control over feasibility -- there is no density knob.
        dim: Sequence length.
        num_motifs: Motifs ``c``.
        motif_length: Tokens per motif, ``k``.
        quantization: Reward levels ``q``, or ``None`` for ``k``.
        epistasis_factor: Coefficient of holo's cubic response. Left at ``0.0``,
            which makes the response the identity, because our landscape has no
            counterpart and a comparison would be measuring its absence.
        random_seed: Seeds the instance.
    """

    num_states: int = 5
    dim: int = 7
    num_motifs: int = 1
    motif_length: int = 3
    quantization: int | None = None
    epistasis_factor: float = 0.0
    random_seed: int = 0

    def as_request(self) -> dict[str, Any]:
        """The parameter dict the worker passes straight to ``Ehrlich``."""
        return {
            "num_states": self.num_states,
            "dim": self.dim,
            "num_motifs": self.num_motifs,
            "motif_length": self.motif_length,
            "quantization": self.quantization,
            "epistasis_factor": self.epistasis_factor,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class ReferenceInstance:
    """A reference Ehrlich instance, pulled across the interpreter boundary.

    Attributes:
        parameters: What it was built from, kept so it can be rebuilt over
            there to score a batch.
        transition_matrix: ``(v, v)`` row-stochastic, zeros marking forbidden
            adjacencies.
        motifs: ``(c, k)`` motif tokens.
        gaps: ``(c, k-1)`` spacings, in holo's own representation -- distances
            between consecutive motif elements, not offsets from the placement.
        optimal_solution: The sequence holo's own ``optimal_solution()``
            constructs and verifies at ``1.0``.
        initial_solutions: ``(n, dim)`` draws from holo's ``initial_solution``.
        optimal_value: holo's declared optimum, ``1.0``.
    """

    parameters: ReferenceParameters
    transition_matrix: npt.NDArray[np.float64]
    motifs: npt.NDArray[np.int64]
    gaps: npt.NDArray[np.int64]
    optimal_solution: npt.NDArray[np.int64]
    initial_solutions: npt.NDArray[np.int64]
    optimal_value: float

    @property
    def offsets(self) -> npt.NDArray[np.int64]:
        """``(c, k)`` cumulative offsets, which is how this package spaces motifs."""
        leading = np.zeros((self.gaps.shape[0], 1), dtype=np.int64)
        return np.concatenate([leading, np.cumsum(self.gaps, axis=1)], axis=1)

    @property
    def transition_density(self) -> float:
        """Fraction of ordered token pairs the chain permits.

        The quantity our generator takes as an argument and holo derives from
        the alphabet size, so it is the first thing a comparison has to check.
        """
        return float(np.mean(self.transition_matrix > 0.0))


def load_reference_instance(
    parameters: ReferenceParameters,
    *,
    interpreter: Path | None = None,
    initial_solutions: int = INITIAL_SOLUTION_POOL,
) -> ReferenceInstance:
    """Construct a reference instance in the holo environment and bring it back.

    Args:
        parameters: What to construct.
        interpreter: Python that can import holo. Located automatically if
            omitted.
        initial_solutions: Wild-type draws to fetch.

    Returns:
        The instance's parameters, as arrays this package can score with.

    Raises:
        HoloUnavailableError: If no holo interpreter exists, or if the worker fails.
    """
    interpreter = require_interpreter() if interpreter is None else interpreter
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        output = workspace / "spec.npz"
        request = workspace / "request.json"
        request.write_text(
            json.dumps(
                {
                    "mode": "spec",
                    "params": parameters.as_request(),
                    "output": str(output),
                    "initial_solutions": initial_solutions,
                }
            ),
            encoding="utf-8",
        )
        _run_worker(interpreter, request)
        with np.load(output) as archive:
            return ReferenceInstance(
                parameters=parameters,
                transition_matrix=archive["transition_matrix"].astype(np.float64),
                motifs=archive["motifs"].astype(np.int64),
                gaps=archive["spacings"].astype(np.int64),
                optimal_solution=archive["optimal_solution"].astype(np.int64),
                initial_solutions=archive["initial_solutions"].astype(np.int64),
                optimal_value=float(archive["optimal_value"]),
            )


def reference_scores(
    parameters: ReferenceParameters,
    sequences: Tokens,
    *,
    interpreter: Path | None = None,
) -> npt.NDArray[np.float64]:
    """Score sequences through holo's own ``evaluate_true``.

    This is the only function that produces reference *values*. It rebuilds the
    instance in the other interpreter rather than shipping the matrices back
    across, so nothing about how we transcribe an instance can influence the
    numbers the reference reports.

    Args:
        parameters: The instance to rebuild and score against.
        sequences: An ``(n, dim)`` array of token indices.
        interpreter: Python that can import holo. Located automatically if
            omitted.

    Returns:
        An ``(n,)`` array, ``-inf`` on infeasible sequences.

    Raises:
        HoloUnavailableError: If no holo interpreter exists, or if the worker fails.
    """
    interpreter = require_interpreter() if interpreter is None else interpreter
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        batch = workspace / "sequences.npy"
        output = workspace / "values.npy"
        request = workspace / "request.json"
        np.save(batch, np.asarray(sequences, dtype=np.int64))
        request.write_text(
            json.dumps(
                {
                    "mode": "score",
                    "params": parameters.as_request(),
                    "sequences": str(batch),
                    "output": str(output),
                }
            ),
            encoding="utf-8",
        )
        _run_worker(interpreter, request)
        values: npt.NDArray[np.float64] = np.load(output).astype(np.float64)
        return values


def _run_worker(interpreter: Path, request: Path) -> None:
    """Execute the worker, turning any failure into one message with its stderr."""
    completed = subprocess.run(  # noqa: S603 - the command is a probed interpreter path
        [str(interpreter), str(_WORKER), str(request)],
        capture_output=True,
        check=False,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        # holo's own errors -- a motif geometry it refuses, a Perron-Frobenius
        # check that fails -- arrive here as a traceback on stderr, and losing
        # it would leave a bare exit code to diagnose from.
        raise HoloUnavailableError(
            f"holo worker failed (exit {completed.returncode}):\n{completed.stderr.strip()}"
        )


class HoloEhrlichLandscape(EhrlichLandscape):
    """A reference Ehrlich instance, scored by this package's evaluator.

    Everything downstream -- the mutation environment, the attainability audit,
    the CMA-ES repairs -- reaches an Ehrlich landscape through the attributes
    installed here, so a reference instance wrapped this way exercises those
    procedures unchanged.

    Args:
        instance: The reference instance to wrap.

    Raises:
        ValueError: If the reference's quantisation does not divide its motif
            length. holo's quantisation formula and ours agree only in that
            case, and scoring such an instance would compare two different
            functions while reporting the difference as a generator discrepancy.
    """

    #: Attributes ``EhrlichLandscape.__init__`` sets and this class must supply
    #: in its place. Public so a test can assert it still matches; a field added
    #: to ``ehrlich.py`` and not here would leave this adapter reading a stale
    #: value with no error anywhere.
    installed_attributes: tuple[str, ...] = (
        "_alphabet",
        "_motif_length",
        "_motifs",
        "_n_motifs",
        "_optimal_sequence",
        "_quantization",
        "_sequence_length",
        "_spacings",
        "_transitions",
    )

    def __init__(self, instance: ReferenceInstance) -> None:
        """Install the reference instance's parameters without generating one."""
        parameters = instance.parameters
        k = parameters.motif_length
        q = k if parameters.quantization is None else parameters.quantization
        if k % q != 0:
            raise ValueError(
                f"holo quantisation q={q} does not divide k={k}; holo would score this instance "
                f"with levels of width k/ceil(k/q)={k / -(-k // q)} and this package with width "
                f"k/q={k / q}, so no disagreement between them would be attributable"
            )

        self._sequence_length = parameters.dim
        self._alphabet = Alphabet.from_string(
            "".join(chr(ord("A") + i) for i in range(parameters.num_states))
            if parameters.num_states <= 26  # noqa: PLR2004 - the Latin alphabet has 26 letters
            else "".join(chr(0x100 + i) for i in range(parameters.num_states))
        )
        self._n_motifs = parameters.num_motifs
        self._motif_length = k
        self._quantization = q
        self._transitions = instance.transition_matrix
        self._motifs = instance.motifs.astype(np.int32)
        self._spacings = instance.offsets.astype(np.int32)
        self._optimal_sequence = instance.optimal_solution.astype(np.int32)
        self._instance = instance

        # The same check `EhrlichLandscape.__init__` makes, and here it is doing
        # more work: it is the first point at which our scoring is applied to
        # holo's motifs, so a mistranscribed spacing convention shows up as a
        # planted optimum that no longer scores 1.0 rather than as a quiet
        # disagreement further down the comparison.
        achieved = float(self.evaluate(self._optimal_sequence[None, :])[0, 0])
        if not np.isclose(achieved, instance.optimal_value):
            raise RuntimeError(
                f"holo's optimal solution scores {achieved} under this package's evaluator, not "
                f"{instance.optimal_value}; the instance was transcribed wrongly"
            )

    @property
    def instance(self) -> ReferenceInstance:
        """The reference instance this landscape wraps."""
        return self._instance

    def feasible_sequence(self, seed: int = 0) -> Tokens:
        """A wild type drawn by holo rather than by us.

        `Task.parent` reaches a campaign's starting sequence through this
        method, and the inherited implementation would walk holo's chain with
        *our* sampler. Returning a draw holo made keeps the whole audit -- wild
        type included -- on the reference side of the comparison.

        Args:
            seed: Selects from the pool of draws fetched with the instance.

        Returns:
            A feasible sequence of the landscape's length.
        """
        pool = self._instance.initial_solutions
        chosen: Tokens = pool[seed % pool.shape[0]].astype(np.int32)
        return chosen
