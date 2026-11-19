# Staggered dynamic-only BC/PPO adaptation

Counts are success/collision/timeout/planning-failure. All training and evaluation episodes have four moving unequal boxes. Motion bounds are18cm Euclidean anchor offset and0.12m/s Euclidean speed; along-route vertical offsets alternate by3-7cm. No static labels or static regression suites are used.

Original source validation: 37/26/1/0. BC warmup selected 0 updates from 200; verified offline teachers: 52 episodes, 3948 transitions.

Actual online motion pairs (amplitude m, speed m/s): [(0.15, 0.1)]. The curriculum did not advance because its dynamic non-regression gates were not met. Offline teachers, fixed-target validation and the enhanced demo use0.18m/0.12m/s.

Training-only source diagnosis: {'success': 110, 'collision': 79, 'timeout': 3}; collision stages: {'middle': 52, 'early': 25, 'terminal': 2}. Contact categories by episode: {'panel / obstacle': 72, 'arm / obstacle': 5, 'panel / wall': 2}. Terminal-query gaps were resolved by22 physical-substep replays, reproducing source outcomes and step counts. No optimizer updates were made during diagnosis.

| Additional PPO steps | Preceding training amplitude/speed | Pure PPO | 50/50 mixture |
|---:|---|---|---|
| 0 | 0.150m / 0.100m/s | 37/26/1/0 | 42/19/1/2 |
| 1000 | 0.150m / 0.100m/s | 36/27/1/0 | 43/18/1/2 |
| 2000 | 0.150m / 0.100m/s | 35/28/1/0 | 42/19/1/2 |
| 3000 | 0.150m / 0.100m/s | 36/27/1/0 | 42/19/1/2 |
| 4000 | 0.150m / 0.100m/s | 37/26/1/0 | 42/19/1/2 |
| 5000 | 0.150m / 0.100m/s | 36/27/1/0 | 42/19/1/2 |

Selected: BC0 plus PPO0. PPO step0 is the validation-selected BC checkpoint. Original source models are preserved. This is one training seed.

Original-source versus final-selected validation pairs: {'success -> success': 37, 'collision -> collision': 26, 'timeout -> timeout': 1}. Rescued seeds: []; lost successes: [].

Independent paired test: not evaluated because no updated checkpoint passed selection; reserved test remains unused.

PPO transitions: 5000; online RRT execution: 5881; online verification replay: 5641. Training outcomes: {'rrt': {'planning_failure': 8, 'success': 71, 'timeout': 1}, 'ppo': {'collision': 62, 'success': 28, 'timeout': 2}}; modes: {'dynamic': 172}. Online auxiliary BC accepted updates: 5. Offline teacher/diagnostic physics are additional and logged separately.

Training failures determine hard-case sampling and teacher phase variants. Evaluation labels never enter either optimizer. All offline/online teacher labels require time checks and independent physical replay. RRT actions never enter the PPO rollout buffer. Scene construction rejects only geometry or unavailable complete nominal references; it never filters by a policy outcome with obstacles. The stronger, staggered scene distribution differs from earlier experiments; compare only the paired source in this report.

See `protocol.json`, `verification.json`, `checkpoints.csv`, `paired_validation.csv`, `curves.png`, and the separate warmup report.
