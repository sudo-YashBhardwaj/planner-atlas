# Amendment 1 to the confirmatory protocol

Recorded 2026-09-20, before any confirmatory branch was trained or evaluated.

The protocol of `runs/confirm/protocol.md` (digest
`b05621fe2abd3966d4129cd99b31265927cb5523db9a1ee8d7efc23878c4ac4c`) is unchanged except for the
two items below, both forced by the protocol-correction pass that came between its registration
and this launch.

1. **Evaluation manifest.** Manifest `86ed349f…` was consumed by the one-seed protocol validation
   run, whose results have been inspected. Confirmation therefore uses a fresh manifest generated
   by exactly the same procedure (48 cases, validation-split episodes only, episode split seed 0,
   validation fraction 0.1) with a new manifest seed. Seeds were tried in order from 303 and the
   first satisfying the protocol's "48 distinct validation-split episodes" requirement was taken,
   which is seed 307; the rule and the seed were fixed before any confirmatory result existed.
   Frozen digest: `5e9458839dccb124ccd94394853a8592b79b27c94a11c4819bbe040dd8b5d746`. It shares no
   case, and one episode, with the consumed manifest.

2. **Seed list.** Seeds 0, 1, 2 are the exploratory grid and seed 3 was observed in the smoke run,
   so the twelve confirmatory seeds are 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, fixed here
   before launch.

Everything else stands: the primary hypothesis, the primary metric (planner-region top-choice
regret), the seed as the unit of replication, the branch set per seed, the frozen training and
planner settings, and the analysis plan. The code is commit `2d6364d`, whose fix pass changed the
latent convention, so held-out losses are not comparable with the exploratory grid.
