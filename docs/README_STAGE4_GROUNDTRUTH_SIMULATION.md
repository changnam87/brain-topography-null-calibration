# Stage 4 ground-truth simulation

## Goal

Address the key reviewer criticism that empirical EEG hyperscanning data do not
provide a known "true synchrony" map.

The simulation is intentionally phase-process based because the primary metric
is PLV. It generates known phase-coupling structure directly, allowing exact
control of true sparse edges and common-drive confounds without introducing a
second arbitrary source model.

## Scenarios

1. `independent_null`
   - empirical 11x40 labels
   - no true edge, no common drive
   - tests false-positive control

2. `global_shared_event`
   - balanced identical labels across groups
   - same event-locked phase drive across groups/participants on positive trials
   - no true inter-brain edge
   - intended to test whether partner-shuffle rejects a globally shared event
     explanation

3. `group_shared_event`
   - empirical labels
   - group-specific common phase drive shared by all three participants
   - no true causal inter-brain edge
   - explicit limitation/stress test: this can mimic partner specificity because
     cross-group shuffling destroys a group-specific common drive

4. `sparse_true`
   - empirical labels
   - exactly 3 known true pair/channel edges on positive trials
   - tests sensitivity and recovery

5. `sparse_true_plus_group_shared`
   - known sparse true edges plus a group-specific common drive
   - tests recovery under a hard confounded condition

## Methods compared

- Raw absolute PLV effect ranking (AUPRC and top-K recovery)
- Naive unadjusted restricted-permutation p < .05
- Label-null global maxT FWER
- Full framework:
    label maxT candidate
    + temporal-shift candidate-family maxT
    + partner-shuffle candidate-family maxT

## Important interpretive boundary

The `group_shared_event` scenario is expected to be difficult because the
implemented temporal and partner nulls cannot, in principle, distinguish every
group-specific common drive from genuine interaction-specific coupling. This is
a feature of the simulation design: it quantifies the framework's boundary
rather than claiming causal proof.

## Workflow

Validate first:

```bash
python3 scripts/28_validate_groundtruth_simulation.py
```

Only after validation and runtime/grid review:

```bash
python3 scripts/29_run_groundtruth_simulation.py
```
