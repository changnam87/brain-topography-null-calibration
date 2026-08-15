# Stage 3 inference freeze

This patch is applied **before viewing any CCC-vs-Other connectivity results**.

## Why the primary statistic changed

Stage 2B6 showed that only 23/44 ten-trial blocks contain both CCC and Other,
and 21/44 are single-class. This means outcome labels are strongly entangled
with session/block/time. A full-triad contrast can therefore mix behavioral
association with block-level drift, learning, fatigue, or other session effects.

The earlier 70% information-retention heuristic is superseded here because it
was a power heuristic, not a validity criterion. No connectivity results had
been viewed when this change was made.

## Frozen primary inference

For each edge/task/band/dyad:

1. Within every informative 10-trial block, compute CCC - Other.
2. Weight that block contrast by:
       n_CCC * n_Other / (n_CCC + n_Other)
3. Aggregate across all triad x block strata.
4. Primary label null: permute CCC/Other labels **within each 10-trial block**,
   preserving block-specific class counts.
5. Multiple-comparison correction: studentized global maxT FWER across all
   primary PLV units.

Secondary:
- triad-stratified information-weighted contrast (no block adjustment)
- equal-triad contrast
- frozen EEG-artifact sensitivity mask

## Why maxT instead of BH on 1,000 empirical permutations

With 9,747 primary units, 1,000 Monte Carlo permutations have minimum attainable
p = 0.000999. That p-value resolution is too coarse for a conventional BH
screen across thousands of units. Studentized maxT directly controls familywise
error from the permutation maximum and avoids arbitrary top-K selection.

## Connectivity implementation

PLV remains the primary metric. Band-pass filtering and the Hilbert transform
are now both computed on an explicitly reflection-padded full released epoch
before cropping the task window. Padding is 2 s by default.

Decision:
  anchor 300 + 0:4 s -> 1200 analysis samples

Feedback:
  anchor 300 + 0:2 s -> 600 analysis samples

Feedback delta remains excluded from primary inference.
