# Dynamic 70-input study: evaluation paused

The user paused formal evaluation after both PPO models completed 1,000,000
training steps and all dynamic validation checkpoints recorded 0/10 successes.
The 50,000-step pilot also completed. Training is complete; evaluation is not.

`episodes.csv` contains 60 completed RRT clean-observation episodes: 55 successes
and zero collisions. No formal PPO test cells have been completed. These partial
results must not be presented as a completed six-cell comparison.

Model validation histories and selected checkpoints are retained in
`models/dynamic70_study/ppo_clean/` and `models/dynamic70_study/ppo_noise/`.
Selected checkpoints are 850,000 and 950,000 steps respectively, chosen by mean
validation reward because validation success was tied at zero throughout.

The separate headless demos at seed 50000 collided for both models; they are not
part of the formal benchmark. Existing data must be preserved if evaluation is
resumed; the original evaluation command deliberately refuses to overwrite this
directory.
