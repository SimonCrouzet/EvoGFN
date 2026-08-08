"""Tests that the baselines package re-exports its constants symmetrically.

The failure is silent and it is a citation failure rather than a runtime one.
``mlde.py`` defines two training sizes that are Wittmann et al.'s
(`PUBLISHED_TRAINING_SIZE`, `PUBLISHED_BUDGET`), one that is a compression of
theirs (`DEFAULT_TRAINING_SIZE`), and one that is **ours** and appears nowhere in
their paper (`ADAPTED_TRAINING_SIZE`, with the `CONSTRAINED_FEASIBLE_FRACTION`
and `CAMPAIGN_PLATES` it is derived from). A package that exports the published
names and hides ours reads as though the published ones were all there is: the
reader who imports from the package sees one set of numbers, the arm registry
runs another, and ``mlde+earlyfit`` -- the row that must never be quoted as MLDE
-- is the row whose defining constant is the missing one.

Nothing raises when that happens. The deep import path keeps working, every test
that reaches past the package keeps passing, and the only symptom is a name that
is harder to find than the one it must be distinguished from.
"""

from evogfn.algorithms import baselines
from evogfn.algorithms.baselines import mlde

#: Every constant in ``mlde.py`` that states a training size or feeds the
#: arithmetic fixing one. Listed rather than derived from the module's public
#: names, because the property under test is that a *chosen* set is exported
#: whole -- a rule that scraped the module would pass by scraping the same
#: omission.
TRAINING_SIZE_CONSTANTS = (
    "PUBLISHED_TRAINING_SIZE",
    "PUBLISHED_BUDGET",
    "DEFAULT_TRAINING_SIZE",
    "ADAPTED_TRAINING_SIZE",
    "CONSTRAINED_FEASIBLE_FRACTION",
    "CAMPAIGN_PLATES",
)


class TestTheTrainingSizesAreExportedTogether:
    """Ours is reachable wherever theirs is, or the asymmetry is the claim."""

    def test_every_training_size_constant_is_re_exported(self):
        missing = [name for name in TRAINING_SIZE_CONSTANTS if not hasattr(baselines, name)]

        assert not missing, (
            f"{', '.join(missing)} is defined in mlde.py but not re-exported, so a reader "
            f"who found PUBLISHED_TRAINING_SIZE through the package would not find it"
        )

    def test_every_re_exported_constant_is_declared_public(self):
        # `__all__` is what a star import and the documentation build read. A
        # name imported into the package but left out of it is exported by
        # accident rather than on purpose, and the next lint pass removes it.
        undeclared = [name for name in TRAINING_SIZE_CONSTANTS if name not in baselines.__all__]

        assert not undeclared, f"{', '.join(undeclared)} is imported but not in __all__"

    def test_the_re_exports_are_the_module_s_own_objects(self):
        # Not copies. Two names holding equal integers would pass every check
        # above and still let the package and the sampler disagree the day one
        # of them moved.
        for name in TRAINING_SIZE_CONSTANTS:
            assert getattr(baselines, name) == getattr(mlde, name)
