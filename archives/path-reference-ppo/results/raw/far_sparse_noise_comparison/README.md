# Paired sparse-obstacle noise comparison

20 paired seeds: 10000–10019. Exactly matching initial joints, true target, and obstacle geometry were asserted across clean/noisy runs.
Three boxes have different sizes and at least 0.15 m surface separation.

| Condition | Success | Collisions | Mean final position error | Mean final orientation error |
|---|---:|---:|---:|---:|
| Clean | 20/20 | 0 | 3.572 mm | 0.078 degrees |
| Noisy target observation | 12/20 | 0 | 10.430 mm | 0.891 degrees |

No planning failures occurred. Noise standard deviations are 5 mm per position axis and 0.5 degrees per rotation-vector axis.
RRT plans once from the initial observation. Success uses the true target (10 mm and 2 degree tolerances).
This is a 20-pair functional comparison, not a population success-rate guarantee; no PPO training was performed.
19 automated tests passed, including paired geometry, unequal dimensions, minimum separation, and physical RRT execution.
Resolved configurations preserve the exact parameters used. Reproduce using scripts/compare_far_noise.py with a new run ID.
