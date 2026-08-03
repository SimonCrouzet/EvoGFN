# Algorithms

One interface, deliberately. A benchmark in which the method under test and the methods it is
compared against run through different harnesses is not a comparison — the budget accounting
drifts, the stopping conditions differ, and the result measures the harness as much as the
method.

::: evogfn.algorithms.base

## GFlowNets

::: evogfn.algorithms.gflownet.sampler

::: evogfn.algorithms.gflownet.training

::: evogfn.algorithms.gflownet.sampling

::: evogfn.algorithms.gflownet.objectives

::: evogfn.algorithms.gflownet.flow_objectives

::: evogfn.algorithms.gflownet.genetic_gfn

::: evogfn.algorithms.gflownet.replay

::: evogfn.algorithms.gflownet.trajectory_balance

## Classical baselines

Directed evolution *is* a genetic algorithm, so these are the incumbents rather than strawmen
to be cleared.

::: evogfn.algorithms.baselines.mutagenesis

::: evogfn.algorithms.baselines.genetic

::: evogfn.algorithms.baselines.directed_evolution

::: evogfn.algorithms.baselines.adalead

::: evogfn.algorithms.baselines.annealing

::: evogfn.algorithms.baselines.cmaes

::: evogfn.algorithms.baselines.mlde

## Giving a baseline the same model access

::: evogfn.algorithms.inner_loop

## The policy network

::: evogfn.models.policy

::: evogfn.models.conditioning
