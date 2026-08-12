#!/usr/bin/env bash
# Tier 1 of manuscript/PLAN.md: finish the replication tier, then add the
# decoder study to it.
#
# Two facts set the shape of this script. Determinism pins every process to one
# thread (`configure_determinism`), so the unit of parallelism has to be the
# process; and the store keeps one file per task and method, so shards never
# contend. Sharding by task alone leaves the two heavy cells --
# `replicate-evolvepro-i7` and `-i5` on the GFlowNet, 14.2 and 6.9 CPU-h -- running
# serially while the rest of the machine idles, which is what `--seed-from` /
# `--seed-to` exist for: every campaign is seeded from its own seed rather than
# from process order, so a sharded run and a serial one produce identical records.
#
# Cost: ~33.6 CPU-h, about 2.8 h wall at 12 workers. The three `+reinit` cells on
# the fixed-anchor tasks are NOT in that figure and will not run -- `_reproduced_on`
# omits them, because a task that never moves its anchor has no move for a carried
# policy to survive.
#
# Safe to re-run: the store resumes, so an interrupted pass picks up where it
# stopped and a completed pass is a no-op.
set -u
cd "/mnt/d/Programmation/GFlowsNets DirectedEvolution"
mkdir -p logs/tier1
WORKERS=${WORKERS:-12}

JOBS=$(mktemp)
trap 'rm -f "$JOBS"' EXIT

# --- T1.0: finish the replication tier, sharded by task x seed range ----------
for t in replicate-alde-i3 replicate-alde-i5 replicate-alde-i7 \
         replicate-evolvepro-i3 replicate-evolvepro-i5 replicate-evolvepro-i7; do
  for lo in 0 25 50 75; do
    hi=$((lo + 25))
    echo ".venv/bin/python experiments/run_suite.py --tier replication --task $t \
--seeds 100 --seed-from $lo --seed-to $hi > logs/tier1/$t.$lo-$hi.log 2>&1" >> "$JOBS"
  done
done

# --- T1.0b: the two short mechanism cells on protocol-evolvepro ---------------
# `+wide` is 6 seeds short. The two `+terminal` rungs are suspended and stay so.
echo ".venv/bin/python experiments/run_suite.py --tier main --task protocol-evolvepro \
--seeds 100 > logs/tier1/protocol-evolvepro.log 2>&1" >> "$JOBS"

# --- T1.1: the decoder study on the tasks that lack it ------------------------
for t in trpb-anchor replicate-alde-i3 replicate-alde-i5 replicate-alde-i7 \
         replicate-evolvepro-i3 replicate-evolvepro-i5 replicate-evolvepro-i7; do
  echo ".venv/bin/python experiments/run_decoder_study.py --task $t \
> logs/tier1/decoder.$t.log 2>&1" >> "$JOBS"
done

echo "$(wc -l < "$JOBS") jobs at $WORKERS workers"
xargs -P "$WORKERS" -I {} bash -c '{}' < "$JOBS"
echo "TIER 1 DONE"
