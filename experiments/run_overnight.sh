#!/usr/bin/env bash
# Two independent bodies of work, run together because they contend for
# nothing: the decomposition rungs write to the result store, and the holo
# sweep writes nothing and lives outside the fingerprinted closure entirely.
set -u
cd "/mnt/d/Programmation/GFlowsNets DirectedEvolution"
mkdir -p logs/overnight
B='gfn-subtb@b0.1-s300-l0.9-h64'

# The holo sweep first and on its own cores: it is the long pole at L=256,
# where the re-anchored audit chains one beam per round.
.venv/bin/python experiments/run_holo_sweep.py --workers 4 \
  > logs/overnight/holo-sweep.log 2>&1 &

# The four decomposition rungs, sharded by task and seed range.
JOBS=$(mktemp); trap 'rm -f "$JOBS"' EXIT
for rung in "+untrained" "+untrained@p2048" "+untrained@p8192" "+untrained@p32768"; do
  for task in protocol-alde protocol-evolvepro feasibility gb1-anchor trpb-anchor; do
    for lo in 0 50; do
      hi=$((lo + 50))
      echo ".venv/bin/python experiments/run_suite.py --tier main --task $task \
--method '${B}${rung}' --seeds 100 --seed-from $lo --seed-to $hi \
> 'logs/overnight/${task}${rung}.${lo}.log' 2>&1" >> "$JOBS"
    done
  done
done
echo "$(wc -l < "$JOBS") rung shards at 11 workers"
xargs -P 11 -I {} bash -c '{}' < "$JOBS"
wait
echo "OVERNIGHT DONE"
