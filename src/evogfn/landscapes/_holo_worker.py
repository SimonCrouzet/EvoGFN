"""Run holo-bench's own ``Ehrlich`` in the interpreter that can import it.

This file is executed as a script by a *different* Python than the one running
the rest of the package, and that separation is the whole reason it exists:
``pytorch-holo`` pins ``numpy<2.0``, which this project cannot adopt without
dragging its entire resolution backwards (see the note in ``pyproject.toml``).
Installing holo beside us is therefore not an option, and reimplementing it
would defeat the purpose of comparing against a reference.

So the contract here is deliberately thin. Nothing in this module may import
anything from ``evogfn``: the holo environment does not have this package
installed, and an import of it would make the bridge fail in a way that looks
like a holo bug. The only wire format is ``.npy``/``.npz``, whose layout has not
changed across the numpy 1/2 boundary for the plain integer and float arrays
used here, and a small JSON request read from ``argv[1]``.

Two modes:

``spec``
    Construct an ``Ehrlich`` and dump everything needed to rebuild it on our
    side -- transition matrix, motifs, spacings, the reference's own
    ``optimal_solution()`` and a pool of ``initial_solution()`` draws.

``score``
    Score an ``(n, dim)`` integer array through the reference's own
    ``evaluate_true``, so the comparison reads holo's arithmetic rather than a
    transcription of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Unresolvable from this project's environment by construction, and the ignore
# is what keeps that from being reported as a mistake every time the package is
# type-checked. A missing holo still fails loudly: the caller runs this file as
# a script and surfaces its stderr, so the ModuleNotFoundError arrives intact.
from holo.test_functions.closed_form import Ehrlich  # type: ignore[import-not-found]


def _build(params: dict[str, Any]) -> Any:  # noqa: ANN401 - holo is unavailable to typing here
    """Construct the reference ``Ehrlich`` from a JSON-decoded parameter dict."""
    return Ehrlich(**params)


def _spec(params: dict[str, Any], output: Path, *, initial_solutions: int) -> None:
    """Dump the reference instance's parameters as an ``.npz`` archive."""
    function = _build(params)
    # `optimal_solution` re-derives and re-checks the planted optimum; letting
    # it raise here is correct, because an instance whose own optimum does not
    # score 1.0 is one no comparison should proceed on.
    optimum = function.optimal_solution()
    # This resets holo's generator to `random_seed`, so the draws are a
    # deterministic function of the instance and not of call order.
    initial = function.initial_solution(n=initial_solutions)

    np.savez(
        output,
        transition_matrix=function.transition_matrix.double().numpy(),
        # Motifs are equal-length, so they stack; spacings are gaps between
        # consecutive motif elements and are one shorter than the motif.
        motifs=torch.stack(list(function.motifs)).to(torch.int64).numpy(),
        spacings=torch.stack(list(function.spacings)).to(torch.int64).numpy(),
        optimal_solution=optimum.to(torch.int64).numpy(),
        initial_solutions=initial.to(torch.int64).numpy().reshape(initial_solutions, -1),
        optimal_value=np.asarray(float(function._optimal_value)),
    )


def _score(params: dict[str, Any], sequences: Path, output: Path) -> None:
    """Score sequences through the reference's own ``evaluate_true``."""
    function = _build(params)
    batch = torch.from_numpy(np.load(sequences).astype(np.int64))
    # `evaluate_true`, not `forward`: the latter casts to float and adds the
    # observation noise, which would make an exact comparison meaningless.
    values = function.evaluate_true(batch)
    np.save(output, values.double().numpy())


def main(argv: list[str]) -> int:
    """Dispatch on the request's ``mode``."""
    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    mode = request["mode"]
    params = request["params"]
    if mode == "spec":
        _spec(params, Path(request["output"]), initial_solutions=request["initial_solutions"])
    elif mode == "score":
        _score(params, Path(request["sequences"]), Path(request["output"]))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
