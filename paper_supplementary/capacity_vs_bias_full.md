# Capacity-vs-bias control on SIFT-1M

Referenced from main paper §4.1. Moved from the main paper to keep within
the PVLDB 12-page limit; the one-paragraph summary in the main paper
captures the headline negative result.

## Setup

All variants are IVF-PQ ($m = 64$, 4-bit, $L = 1000$, $n_p = 20$) evaluated
on the full 1M database with 1,000 queries; ground truth recomputed against
the 1M database. Reproduced by `experiments/streaming_capacity_vs_bias.py`.

Three variants disambiguate whether the streaming drop is a
codebook-*bias* artefact (the initial 200K is unrepresentative) or a
codebook-*capacity* artefact (a 200K-trained codebook is too small for
the eventual 1M database, regardless of which 200K is used):

- **A. PQ-200K-initial** — codebook+partition fitted on the first 200K of the stream (the "stale" condition).
- **B. PQ-200K-random** — codebook+partition fitted on a uniformly random 200K sample of the full 1M (same training size as A, no initial-sample bias).
- **C. PQ-1M** — codebook+partition fitted on the full 1M (oracle, no bias and no capacity gap).

The **B − A** gap isolates initial-sample bias; the **C − B** gap isolates the
residual capacity contribution at this bit budget.

## Results

| Variant | Training sample | R@10 on 1M | Notes |
|---|---|---|---|
| A. PQ-200K-initial | first 200K of stream | 71.56% | streaming "stale" |
| B. PQ-200K-random  | random 200K of 1M    | 71.31% | same size, no bias |
| C. PQ-1M           | full 1M (oracle)     | 71.16% | no bias, no capacity gap |
| **Bias contribution** (B − A)     | | | **−0.25 pp** |
| **Capacity contribution** (C − B) | | | **−0.15 pp** |

## Reading

Both gaps (B − A and C − B) are within statistical noise (SE ≈ 1.4 pp on
1,000 queries at ~71% recall): the spread A → B → C is 0.40 pp, well below
2 SE. The defensible reading is the weaker one: **at this bit budget, the
choice of which 200K is used to train the codebook does not detectably
matter.** Jointly with the 10M retrain experiments in main paper §4.2 (where
re-training every batch costs hundreds of seconds and recovers ≤ 0.25 pp),
the negative result is that **retraining the codebook is not the lever
that closes the streaming gap.**

The companion three-condition shift control
([`table3_streaming_controls.md`](table3_streaming_controls.md)) uses a
looser search budget ($L = 500$, $n_p = 10$, 10K queries); the two operating
points are not directly comparable in absolute recall, but the *spread*
between codebook variants at fixed protocol is the quantity each table
isolates.
