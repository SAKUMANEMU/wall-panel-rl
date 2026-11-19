# PROJECT MASTER INSTRUCTION

You are responsible for implementing, testing, debugging, documenting, and evaluating an entire machine-learning course project:

**Reinforcement Learning for Collision-Aware Robotic Wall Panel Installation**

Do not only generate example code or a skeleton repository. Build a complete runnable project, execute the important scripts, fix errors you encounter, verify that the environment works, train at least basic PPO models, run the classical baseline, generate evaluation results and plots, and prepare the repository so that another person can reproduce the project from `README.md` and `environment.yml`.

Work incrementally. After each implementation phase, run the relevant tests/demo before moving to the next phase.

Do not silently change the task formulation. If a technical change is necessary, document it in `DECISIONS.md`.

---

# 1. PROJECT OBJECTIVE

The task is robotic wall-panel manipulation.

A robotic manipulator already grasps a small rigid panel. It must move the panel from a randomized initial configuration to a predefined target installation pose while:

- avoiding collisions with obstacles;
- avoiding robot self-collision where applicable;
- respecting joint limits;
- accurately matching the target panel position and orientation;
- minimizing unnecessary motion;
- remaining robust when the observed target pose contains measurement noise.

The primary machine-learning method is:

**PPO with joint-space incremental actions.**

The primary conventional baseline is:

**RRT-Connect joint-space motion planning + joint-space feedback/position control.**

The main scientific comparison is:

**PPO learned reactive control vs classical collision-aware planning and control, under noise-free and noisy pose observations.**

Do NOT include observation delay in this project.

---

# 2. IMPORTANT DESIGN DECISIONS

These decisions should be treated as fixed unless a serious implementation issue makes one impossible.

## 2.1 Simulator

Use:

- Python
- PyBullet
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- SciPy where useful
- Pandas
- Matplotlib
- PyYAML

PyBullet is chosen because the course project should focus on reinforcement learning and obstacle avoidance rather than ROS/Gazebo integration complexity.

The environment must work in:

- GUI mode for demonstration;
- DIRECT/headless mode for training.

GPU usage should be optional.

The project must still work on CPU.

---

# 3. ROBOT

Use a standard articulated manipulator suitable for the task.

Prefer a 6-DOF robot such as UR5/UR5e.

If a reliable UR5/UR5e URDF is not readily available in the working environment, use another standard manipulator but clearly document the change.

If a non-6-DOF robot is used, the RL action dimension must equal the number of controllable arm joints.

Do not fake robot kinematics.

The robot must have:

- valid revolute joints;
- known joint limits;
- end-effector link;
- collision geometry;
- forward kinematics available through the simulator.

Place all robot/model assets inside:

`assets/`

Document the source and license/attribution of any external URDF/model.

---

# 4. PANEL MODEL

The panel is already grasped by the robot.

Start with a deliberately small panel to verify feasibility.

Recommended initial dimensions:

- width: 0.30 m
- height: 0.20 m
- thickness: 0.03 m

The exact dimensions must be stored in configuration files and not hard-coded.

Attach the panel rigidly to the robot end-effector using an appropriate fixed constraint.

The panel collision geometry must be included in collision checking.

Later experiments may optionally use a larger panel, but the small panel is the required starting point.

---

# 5. TARGET INSTALLATION POSE

Create a predefined target panel pose.

Represent it using:

- target position;
- target orientation quaternion.

A wall may be displayed for visual context.

However, do not require physical wall contact for success because contact dynamics can introduce unnecessary instability.

Instead, define the target pose slightly in front of the wall so that a successful panel pose is collision-free.

For example, leave approximately 1–3 cm clearance between the target panel and the wall.

The final target should be realistically reachable.

---

# 6. OBSTACLES

Start with sparse obstacles.

Do not create a cluttered environment initially.

Use simple axis-aligned or oriented boxes.

Required curriculum:

### Stage A
No obstacles.

### Stage B
One sparse obstacle.

### Stage C
1–3 sparse randomized obstacles.

Obstacle properties must be configurable:

- position;
- dimensions;
- number of obstacles;
- randomization range.

Obstacle placement must satisfy:

- no overlap with initial robot/panel;
- no overlap with target panel;
- sparse enough that feasible paths normally exist.

Avoid generating impossible environments during training.

Create deterministic evaluation scenarios using fixed random seeds.

---

# 7. GYMNASIUM ENVIRONMENT

Implement a custom environment, preferably:

`WallPanelEnv`

It must follow the Gymnasium API.

Required methods include:

- `__init__`
- `reset`
- `step`
- `render`
- `close`

Use deterministic seeding correctly.

The environment should support both:

`render_mode=None`

and

`render_mode="human"`

Training should use PyBullet DIRECT mode.

Demo should use GUI mode.

---

# 8. OBSERVATION / STATE SPACE

Use a fixed-size continuous observation vector.

At minimum include:

## Robot state

Joint positions:

`q`

Joint velocities:

`dq`

## Target information

Relative position error:

`target_position - panel_position`

Use three values.

Relative orientation error:

use a stable representation such as a 3D axis-angle / rotation-vector error.

Do NOT directly use raw Euler angle subtraction if avoidable.

## Obstacle information

Define a maximum number of obstacles:

`MAX_OBSTACLES`

Recommended initial value:

`3`

For each obstacle include something similar to:

- relative center x;
- relative center y;
- relative center z;
- size x;
- size y;
- size z;
- valid/presence mask.

Zero-pad unused obstacle slots.

Prefer obstacle position relative to the current panel/end-effector rather than absolute world coordinates.

All observation dimensions must remain constant.

Create helper functions for normalization.

Avoid extremely different numerical scales.

Do not include observation delay.

---

# 9. OBSERVATION NOISE

Measurement noise is one of the primary experimental variables.

The true simulator state must remain noise-free internally.

Noise should only affect what the controller/agent observes.

At minimum support:

## Noise-free

position noise:

`σ_position = 0`

orientation noise:

`σ_orientation = 0`

## Noisy

Provide configurable Gaussian noise.

Reasonable initial values:

position standard deviation:

`0.005 m`

orientation standard deviation:

`0.5 degrees`

Store all values in YAML config.

Do not hard-code them.

Optionally support additional noise levels:

- low;
- medium;
- high.

But the primary required comparison is:

**no noise vs noise.**

Evaluation metrics must always be calculated using the ground-truth simulator pose, not the noisy observation.

---

# 10. PPO ACTION SPACE

Do NOT use Cartesian end-effector actions.

The TA recommendation must be followed.

Use incremental joint-angle commands.

For an n-joint arm:

`action ∈ [-1, 1]^n`

Convert normalized actions into joint increments:

`delta_q = action * max_joint_increment`

Then:

`q_target = q_current + delta_q`

Clamp or validate according to joint limits.

Recommended starting maximum increment:

approximately:

`2–5 degrees per action`

Make this configurable.

The action represents a new local joint target.

Use the simulator's joint position controller as the common low-level actuator interface.

The PPO policy should therefore learn:

- how to move toward the installation pose;
- how to select robot configurations;
- how to move around obstacles;
- how large each correction should be.

---

# 11. COMMON LOW-LEVEL ACTUATOR

For fairness, PPO and the classical planner should use the same low-level robot actuator interface.

Implement something similar to:

`JointPositionController`

Input:

`desired joint configuration q_des`

Output:

joint motor commands.

PyBullet `POSITION_CONTROL` is acceptable.

Configure position/velocity gains in YAML.

Both:

- PPO;
- RRT-Connect baseline

must ultimately command desired joint positions through this same interface.

Do not let one method use unrealistic instantaneous joint teleportation while the other uses motor control.

`resetJointState` may be used internally for collision checking in the planner, but not as the actual execution method.

---

# 12. COLLISION CHECKING

Create a centralized collision checking module.

It should support:

- robot vs obstacles;
- panel vs obstacles;
- robot vs wall if relevant;
- panel vs wall except permitted target clearance;
- self-collision where supported;
- joint limit validation.

Create functions similar to:

`is_configuration_valid(q)`

and:

`is_collision()`

The planner and RL environment must use the same collision definitions whenever possible.

This is important for experimental fairness.

---

# 13. COLLISION RESPONSE DURING RL

During training, a collision should strongly penalize the policy.

Recommended default:

- large negative collision reward;
- terminate the episode.

Example:

`reward -= 50`

then:

`terminated = True`

Do not assume these exact values are optimal.

Place them in configuration.

Joint-limit violations should similarly receive a significant penalty.

Avoid silently clipping every invalid action without informing the learning algorithm.

---

# 14. SUCCESS CONDITION

An episode succeeds only when the ground-truth panel pose satisfies both:

position error:

`||p_target - p_panel|| < position_tolerance`

and orientation error:

`orientation_error < orientation_tolerance`

Suggested initial tolerances:

position:

`0.01 m`

orientation:

`2 degrees`

Later they may be tightened.

Make them configurable.

Successful completion should provide a large positive terminal reward.

---

# 15. REWARD FUNCTION

Use a progress-based reward rather than only absolute distance.

Define:

`d_pos(t)`

as current position error.

Define:

`d_ori(t)`

as current orientation angular error.

Use terms such as:

`position_progress = d_pos(t-1) - d_pos(t)`

`orientation_progress = d_ori(t-1) - d_ori(t)`

A suitable structure is:

`reward =`
`w_pos * position_progress`
`+ w_ori * orientation_progress`
`- w_action * ||action||²`
`- step_penalty`

Terminal additions:

collision:

large negative reward

joint-limit violation:

large negative reward

success:

large positive reward

Optional:

small penalty for unnecessarily large joint motion.

Do NOT reward direct distance reduction so strongly that the agent chooses to collide through an obstacle.

All weights must be configured through YAML.

Log individual reward components for debugging.

---

# 16. PPO IMPLEMENTATION

Use Stable-Baselines3 PPO.

Use an MLP actor-critic architecture.

Start with:

- 2 hidden layers;
- approximately 128 neurons each;
- ReLU.

Use:

`MlpPolicy`

unless environment design requires another policy.

Create a configurable training script.

For example:

`scripts/train_ppo.py`

Support arguments such as:

`--config`

`--seed`

`--total-timesteps`

`--run-name`

`--device`

`--render`

Save:

- model checkpoints;
- best model;
- final model;
- VecNormalize statistics if used;
- training configuration;
- random seed;
- TensorBoard logs.

Use evaluation callbacks.

Do not overwrite previous runs.

---

# 17. PPO TRAINING CONDITIONS

Train at least two meaningful variants.

## PPO-Clean

Train with:

- no observation noise.

## PPO-Noise

Train using configured pose measurement noise or randomized noise.

The purpose is to test whether noise-aware training improves robustness.

If compute resources are insufficient for full training, still implement both variants and run enough training to verify learning trends.

Do not pretend an insufficiently trained model has converged.

Clearly report training length.

---

# 18. CLASSICAL BASELINE

The professor specifically questioned whether collision constraints merely reject infeasible actions.

The main classical baseline must therefore actively generate a collision-free path around obstacles.

Implement:

**RRT-Connect in joint configuration space.**

Do not use only:

`move directly toward target -> collision -> reject`

as the main baseline.

---

# 19. RRT-CONNECT GOAL CONFIGURATION

Given the target panel pose:

1. convert the desired panel pose into the corresponding desired end-effector pose;
2. use inverse kinematics to obtain one or more candidate goal joint configurations;
3. reject candidates violating:
   - joint limits;
   - collision constraints;
4. choose a valid goal configuration.

Record IK failure explicitly.

If several IK seeds are useful, try multiple seeds.

Do not compare RL "IK failure rate" directly because RL actions no longer use IK.

Instead report classical:

- IK goal failure rate;
- planning failure rate.

---

# 20. RRT-CONNECT IMPLEMENTATION

Implement a clear joint-space RRT-Connect algorithm.

Prefer implementing it within the project rather than hiding the algorithm inside an opaque external planner.

Core features:

- random configuration sampling;
- goal bias;
- nearest-node search;
- bounded extension;
- two trees;
- tree connection;
- collision checking;
- interpolated edge validation;
- maximum iteration count;
- timeout;
- path extraction;
- optional shortcut smoothing.

Important configuration parameters:

- step size;
- collision interpolation resolution;
- maximum iterations;
- goal bias;
- planning timeout.

Suggested starting values may be chosen and then tuned.

All parameters go in YAML.

Create unit tests for core planner helpers where practical.

---

# 21. CLASSICAL EXECUTION

After RRT-Connect returns:

`q0, q1, ..., qN`

interpolate/smooth the path.

Execute the resulting waypoints through the same `JointPositionController` used by PPO.

Record:

- planning success;
- planning time;
- execution time;
- collision status;
- target accuracy;
- path length;
- number of executed control steps.

---

# 22. OPTIONAL DIAGNOSTIC BASELINE

Implement an optional simple baseline:

**Direct goal-seeking controller + collision rejection**

This controller may move toward the target but reject a step if the resulting configuration collides.

This is NOT the main conventional baseline.

Its purpose is to demonstrate the professor's concern:

collision rejection alone does not necessarily produce a path around an obstacle.

Label it clearly:

`RejectOnlyBaseline`

Use it only as an ablation/diagnostic comparison.

---

# 23. EXPERIMENTAL CURRICULUM

Do not begin with the hardest task.

Implement and verify the project in this order.

## Phase 1

Small panel.

No obstacle.

No noise.

Goal:

confirm robot control, environment, reward, PPO interface, target pose calculation and success detection.

## Phase 2

Small panel.

One fixed sparse obstacle.

No noise.

Goal:

confirm collision detection and RRT-Connect.

## Phase 3

Small panel.

Random sparse obstacles.

No noise.

Goal:

test PPO collision-aware behavior and generalization.

## Phase 4

Random sparse obstacles.

Noise-free vs noisy observations.

Goal:

main PPO vs classical baseline experiment.

## Optional Phase 5

Larger panel.

Goal:

investigate how increased panel collision geometry affects task difficulty.

Do not attempt Phase 5 until Phases 1–4 work.

---

# 24. EVALUATION PROTOCOL

Create a separate evaluation script.

For example:

`scripts/evaluate.py`

Evaluation must use fixed seeds.

Use the same scenario set for all methods.

Each scenario should define:

- initial robot configuration;
- obstacle configuration;
- target pose;
- observation noise seed.

Evaluate at least:

- PPO-Clean;
- PPO-Noise;
- RRT-Connect + controller.

Optional:

- RejectOnlyBaseline.

Use enough episodes to estimate success and collision rates.

Aim for at least 100 evaluation episodes per major condition if computationally practical.

If fewer episodes are used, state the exact number.

---

# 25. PRIMARY EXPERIMENT MATRIX

At minimum generate results for:

| Method | Observation |
|---|---|
| RRT-Connect + controller | No noise |
| PPO-Clean | No noise |
| PPO-Noise | No noise |
| RRT-Connect + controller | Noise |
| PPO-Clean | Noise |
| PPO-Noise | Noise |

Optional additional noise sweep:

- 0 mm / 0°;
- 2 mm / 0.2°;
- 5 mm / 0.5°;
- 10 mm / 1.0°.

Do not add this sweep until the primary experiment works.

---

# 26. EVALUATION METRICS

Calculate and save at least:

1. installation success rate;
2. collision rate;
3. final position error in meters;
4. final orientation error in degrees;
5. episode/control steps;
6. joint-space path length;
7. end-effector/panel Cartesian path length;
8. execution time;
9. planning time for RRT;
10. RRT planning failure rate;
11. target IK failure rate for the classical method;
12. mean cumulative reward for PPO where useful.

Report mean and standard deviation for continuous metrics.

For success/collision metrics report percentages.

Do not use noisy measurements to calculate final accuracy.

Use ground truth.

---

# 27. FAIRNESS REQUIREMENTS

The comparison must be defensible.

Ensure:

- same robot;
- same panel;
- same obstacle geometry;
- same initial configurations;
- same target;
- same collision definitions;
- same low-level joint actuator;
- same evaluation scenarios;
- same observation noise realization when appropriate.

The methods differ primarily in high-level decision making:

PPO:

`observation -> neural policy -> incremental joint target`

Classical:

`observed target -> IK -> RRT-Connect -> joint trajectory`

Document these differences explicitly.

---

# 28. NOISE AND CLASSICAL PLANNER

Be explicit about how observation noise affects the classical method.

The planner/controller must not secretly access perfect target measurements if PPO receives noisy target observations.

Use the same noisy observed target pose as input to target estimation.

However:

- simulator collision geometry;
- true robot joint state;
- ground-truth evaluation

may remain exact unless otherwise specified.

Clearly document which variables are noisy and which are ground truth.

---

# 29. VISUALIZATION

The course project requires graphical visualization.

Create:

`scripts/demo.py`

It must launch PyBullet GUI and support commands such as:

`python scripts/demo.py --method ppo`

and:

`python scripts/demo.py --method rrt`

and optionally:

`python scripts/demo.py --method reject`

The visualization should show:

- robot;
- panel;
- obstacles;
- target pose;
- optional wall;
- current trajectory.

Use a visually distinct transparent or wireframe marker for target panel pose.

During demo, print or overlay:

- current position error;
- orientation error;
- step number;
- collision state;
- success state.

Make the demo deterministic by default using a known seed.

---

# 30. RESULT PLOTS

Automatically generate publication/report-quality plots using Matplotlib.

Required plots should include:

## Training

- PPO training reward vs steps;
- PPO evaluation success rate vs training steps if available.

## Main comparison

- success rate by method and noise condition;
- collision rate by method and noise condition;
- final position error;
- final orientation error;
- number of steps;
- execution/planning time where appropriate.

## Trajectory example

For representative episodes, plot:

- end-effector/panel 3D trajectory;
- target;
- obstacle positions.

Optionally plot joint trajectories.

All plots must have:

- clear title;
- axis labels;
- units;
- legend;
- readable fonts.

Save plots under:

`results/figures/`

Do not manually fabricate values.

Plots must come from generated CSV evaluation results.

---

# 31. DATA OUTPUT

Every evaluation episode should produce a structured row.

Store results in CSV.

Suggested columns:

- run_id
- method
- training_condition
- evaluation_noise
- seed
- success
- collision
- planning_success
- ik_success
- position_error_m
- orientation_error_deg
- steps
- joint_path_length_rad
- cartesian_path_length_m
- planning_time_s
- execution_time_s
- cumulative_reward

Create:

`results/raw/`

and:

`results/summary/`

Write a summarization script that calculates grouped statistics.

---

# 32. CONFIGURATION SYSTEM

Use YAML configuration files.

Recommended:

`configs/base.yaml`

`configs/train_clean.yaml`

`configs/train_noise.yaml`

`configs/eval_clean.yaml`

`configs/eval_noise.yaml`

Store:

- simulator timestep;
- robot parameters;
- panel dimensions;
- target tolerances;
- obstacle parameters;
- noise levels;
- reward coefficients;
- PPO hyperparameters;
- action scaling;
- controller gains;
- RRT parameters;
- random seeds.

Avoid duplicated magic numbers throughout the code.

---

# 33. REPOSITORY STRUCTURE

Use a clean structure similar to:

```
project_root/
│
├── README.md
├── environment.yml
├── DECISIONS.md
├── REPORT_DRAFT.md
├── .gitignore
│
├── assets/
│   ├── robot/
│   └── README.md
│
├── configs/
│   ├── base.yaml
│   ├── train_clean.yaml
│   ├── train_noise.yaml
│   ├── eval_clean.yaml
│   └── eval_noise.yaml
│
├── src/
│   └── wall_panel_rl/
│       ├── __init__.py
│       │
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── wall_panel_env.py
│       │   ├── collision.py
│       │   ├── geometry.py
│       │   └── noise.py
│       │
│       ├── control/
│       │   ├── __init__.py
│       │   └── joint_controller.py
│       │
│       ├── planning/
│       │   ├── __init__.py
│       │   ├── rrt_connect.py
│       │   ├── ik.py
│       │   └── path_utils.py
│       │
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── classical.py
│       │   └── reject_only.py
│       │
│       ├── rl/
│       │   ├── __init__.py
│       │   ├── callbacks.py
│       │   └── policies.py
│       │
│       └── utils/
│           ├── config.py
│           ├── metrics.py
│           ├── seeding.py
│           └── plotting.py
│
├── scripts/
│   ├── smoke_test.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── demo.py
│   └── plot_results.py
│
├── tests/
│   ├── test_environment.py
│   ├── test_collision.py
│   ├── test_rrt.py
│   ├── test_reward.py
│   └── test_determinism.py
│
├── models/
│
├── logs/
│
└── results/
    ├── raw/
    ├── summary/
    └── figures/
```

Modify structure only if necessary.

---

# 34. ENVIRONMENT.YML

Create an exact reproducible Conda environment.

The course requires exact Python/library versions.

Do NOT invent version numbers without testing them.

Install and test a compatible environment and then pin the versions actually used.

At minimum include:

- Python
- numpy
- scipy
- pandas
- matplotlib
- pyyaml
- pybullet
- gymnasium
- stable-baselines3
- torch
- tensorboard
- pytest

If pip packages are required inside Conda YAML, list them correctly under:

`pip:`

The environment should be creatable using:

`conda env create -f environment.yml`

Document the environment name.

---

# 35. README

Write a complete README.

Someone unfamiliar with the code should be able to run the project.

Include:

## Project overview

Explain the robotic panel task.

## Method

Explain:

- PPO;
- joint-space actions;
- RRT-Connect baseline;
- observation noise experiment.

## Installation

Exact commands.

## Smoke test

Example:

`python scripts/smoke_test.py`

## Training

Commands for clean PPO and noise PPO.

## Evaluation

Commands.

## Demo

Commands for PPO and RRT.

## Plot generation

Commands.

## Repository structure

Short explanation.

## Reproducibility

Explain random seeds.

## External resources

Disclose:

- PyBullet;
- Stable-Baselines3;
- Gymnasium;
- robot URDF/model;
- any other reused libraries/code.

Do not imply external algorithms/code were written from scratch if they were not.

---

# 36. REPORT_DRAFT.MD

Prepare a report draft as part of the project.

It should contain at least:

# 1. Introduction

Motivation for robotic wall-panel installation.

# 2. Problem Formulation

State, action, reward, success and collision definitions.

# 3. Code Explanation

Explain the organization and main logic.

# 4. Existing Codes / Libraries / External Tools

Explicitly disclose:

- PyBullet;
- Gymnasium;
- Stable-Baselines3;
- PyTorch;
- robot model;
- any external code.

# 5. Conventional Baseline

Explain why collision rejection alone is insufficient.

Explain:

RRT-Connect + joint-space controller.

# 6. Reinforcement Learning Method

Explain PPO and network architecture.

# 7. Simulation Setup

Explain:

- robot;
- small panel;
- sparse obstacles;
- no delay;
- observation noise.

# 8. Experimental Design

Explain noise-free/noisy comparison.

# 9. Metrics

List all metrics.

# 10. Results

Populate automatically or manually from real experiment outputs.

Never invent experimental numbers.

# 11. Discussion

Discuss:

- PPO strengths;
- classical planning strengths;
- noise effects;
- collision avoidance;
- failure cases.

# 12. Limitations

Examples:

- simplified panel;
- sparse obstacles;
- simulator-only;
- limited training;
- no real communication/control delay;
- simplified sensing.

# 13. Conclusion

Summarize findings supported by actual results.

---

# 37. TESTING

Create real tests.

At minimum verify:

## Environment

`reset()` returns correct observation shape.

`step()` works.

Observation is finite.

Action dimensions are correct.

## Joint limits

Actions cannot silently produce invalid states.

## Collision

Known colliding configurations are detected.

Known collision-free configurations are accepted.

## Reward

Moving toward target should usually produce greater progress reward than moving away, in controlled unit-test scenarios.

## RRT

Planner can solve at least one simple collision-free problem.

Planner avoids a simple obstacle.

## Determinism

Same seed should reproduce environment randomization where practical.

Use:

`pytest`

and ensure tests pass before final completion.

---

# 38. SMOKE TEST

Create:

`scripts/smoke_test.py`

It should run quickly.

It must verify:

1. imports;
2. simulator launch in DIRECT mode;
3. robot loading;
4. panel attachment;
5. environment reset;
6. random actions;
7. collision checking;
8. target error calculation;
9. one basic RRT planning call;
10. one short PPO environment compatibility check.

The smoke test should finish without training a full model.

README should instruct the grader to run this first.

---

# 39. GYMNASIUM / SB3 VALIDATION

Use Stable-Baselines3 environment checker.

Run:

`check_env`

during development.

Resolve warnings that indicate real API problems.

Make sure:

- observation dtype matches observation space;
- action dtype/shape is valid;
- terminated and truncated follow Gymnasium API correctly;
- reset returns `(obs, info)`;
- step returns `(obs, reward, terminated, truncated, info)`.

---

# 40. INFORMATION RETURNED IN `info`

Every environment step should include useful debugging data in `info`, such as:

- position_error;
- orientation_error;
- collision;
- success;
- joint_limit_violation;
- reward_position_progress;
- reward_orientation_progress;
- reward_action;
- reward_step;
- true_panel_pose;
- observed_target_error where useful.

This will simplify evaluation and debugging.

---

# 41. TERMINATION VS TRUNCATION

Use:

`terminated=True`

for:

- success;
- collision;
- severe invalid state if appropriate.

Use:

`truncated=True`

for:

- reaching maximum episode length.

Do not mix these concepts.

---

# 42. PATH LENGTH

For PPO and baseline compute comparable path metrics.

Joint-space path length:

sum over:

`||q[t+1] - q[t]||`

Cartesian path length:

sum over panel/end-effector position displacement.

Do not only compare number of actions because planner waypoint frequency may differ.

---

# 43. TRAINING DEBUGGING PROCEDURE

If PPO does not learn, debug systematically.

Do NOT immediately make the task easier without understanding the issue.

Check:

1. can the robot reach the target?
2. does the success detector work?
3. is the reward sign correct?
4. are observations normalized?
5. are actions too small or too large?
6. are episodes terminating immediately?
7. are random initial states valid?
8. can a scripted policy decrease distance?
9. can PPO solve Phase 1 with no obstacles?
10. only then add obstacles.

Create diagnostic plots/logs where useful.

Document major changes in `DECISIONS.md`.

---

# 44. CURRICULUM TRAINING

If PPO struggles with random obstacles, implement curriculum gradually.

Possible training progression:

### Curriculum 0
No obstacle.

### Curriculum 1
One fixed obstacle.

### Curriculum 2
One randomized obstacle.

### Curriculum 3
1–3 randomized obstacles.

Do not introduce complex curriculum logic unless needed.

Keep configuration reproducible.

---

# 45. TRAINING WITH NOISE

For the noise-trained model, noise may be randomized per episode.

For example:

translation noise standard deviation sampled from:

`0 to 5 mm`

orientation noise standard deviation sampled from:

`0 to 0.5 degrees`

This acts as domain randomization.

But keep a simple fixed-noise option as well.

Record exactly what is used.

---

# 46. IMPORTANT SCIENTIFIC DISTINCTION

Do not claim:

"joint-space action automatically guarantees collision avoidance."

That is incorrect.

Joint-space actions allow the policy to reason directly over robot configuration, but collision avoidance still emerges through:

- state information;
- reward;
- collision penalties;
- training;
- collision-aware environment interaction.

Likewise, RRT achieves collision avoidance because sampled configurations and edges are explicitly collision checked.

This distinction should appear in the report.

---

# 47. IMPORTANT BASELINE DISTINCTION

Clearly distinguish:

## Collision checking

Determines whether a configuration/path is valid.

from:

## Collision avoidance / path planning

Actively generates an alternative path around an obstacle.

A controller that only rejects colliding moves is not equivalent to RRT-based motion planning.

This point should directly answer the professor's concern.

---

# 48. CODE QUALITY

Use:

- type hints where helpful;
- docstrings;
- descriptive variables;
- comments explaining non-obvious robotics/RL logic;
- small functions;
- modular design.

Do not create a giant 1000-line environment file.

Avoid unnecessary abstraction.

Code should be understandable to a course instructor.

---

# 49. DO NOT USE NOTEBOOKS AS THE PRIMARY IMPLEMENTATION

Jupyter notebooks may optionally be created for exploration.

However all required functionality must be runnable using normal Python scripts.

The grader should not need a notebook to:

- train;
- evaluate;
- plot;
- demo.

---

# 50. REPRODUCIBILITY

Seed:

- Python random;
- NumPy;
- Gymnasium;
- PyTorch;
- Stable-Baselines3 environment;
- obstacle generator.

Store evaluation scenario seeds.

Print the config and seed at the start of each experiment.

Save a copy of the resolved config with each model/run.

---

# 51. GIT

Use sensible commits if Git access is available.

Examples:

- initialize environment and simulator;
- add WallPanelEnv;
- implement collision checking;
- implement RRT-Connect;
- add PPO training;
- add evaluation pipeline;
- add visualization;
- add experiment plots;
- finalize documentation.

Do not commit:

- huge TensorBoard logs;
- temporary files;
- large checkpoints unless needed;
- Python caches.

Create an appropriate `.gitignore`.

---

# 52. REQUIRED FINAL COMMANDS

Aim for the repository to support commands conceptually similar to:

```bash
conda env create -f environment.yml
conda activate wall-panel-rl

python scripts/smoke_test.py

pytest -q

python scripts/train_ppo.py \
    --config configs/train_clean.yaml

python scripts/train_ppo.py \
    --config configs/train_noise.yaml

python scripts/evaluate.py \
    --config configs/eval_clean.yaml

python scripts/evaluate.py \
    --config configs/eval_noise.yaml

python scripts/plot_results.py

python scripts/demo.py \
    --method rrt

python scripts/demo.py \
    --method ppo \
    --model models/<MODEL_NAME>.zip
```

Adjust exact arguments as necessary but preserve this level of usability.

---

# 53. ACCEPTANCE CRITERIA

Do not consider the project complete until the following are true.

### Installation

`environment.yml` is present and valid.

### Environment

The custom Gymnasium environment runs.

### Simulation

Robot, panel, target and obstacles are visible in GUI.

### Joint-space PPO

Action is incremental joint-space control, not Cartesian control.

### Collision

Panel and robot collision checking works.

### RRT baseline

RRT-Connect can plan around at least a representative sparse obstacle.

### PPO

PPO training script runs and produces a saved model.

### Noise

Noise-free and noisy observation modes both work.

### Evaluation

The same scenario set can evaluate PPO and RRT.

### Metrics

Results are saved to CSV.

### Figures

Evaluation plots can be generated from CSV.

### Demo

A graphical demo works.

### Tests

Core pytest tests pass.

### Documentation

README is sufficient for another user to run the project.

### Report

`REPORT_DRAFT.md` accurately explains the implemented project.

---

# 54. DEVELOPMENT ORDER

Implement the project in this exact broad order unless there is a strong reason to change it.

## Step 1
Create repository structure and Conda environment.

## Step 2
Load robot and create simple simulation.

## Step 3
Attach small panel.

## Step 4
Implement target pose and pose-error calculation.

## Step 5
Implement joint position actuator.

## Step 6
Implement Gymnasium environment without obstacles.

## Step 7
Verify random/scripted actions and success condition.

## Step 8
Implement collision checker.

## Step 9
Add one obstacle.

## Step 10
Implement RRT-Connect.

## Step 11
Verify classical baseline visually.

## Step 12
Implement PPO interface.

## Step 13
Verify PPO can learn no-obstacle task.

## Step 14
Add obstacle curriculum.

## Step 15
Add observation noise.

## Step 16
Train clean/noise PPO models.

## Step 17
Create fixed benchmark scenarios.

## Step 18
Run PPO/RRT experiments.

## Step 19
Generate CSV summaries and figures.

## Step 20
Create demo and finalize documentation/report.

Do not jump directly to long PPO training before Steps 1–13 are verified.

---

# 55. WHEN SOMETHING FAILS

You are expected to debug the implementation.

If an error occurs:

1. inspect the traceback;
2. identify the actual cause;
3. fix it;
4. rerun the failing command;
5. verify no regression.

Do not simply comment out failing functionality.

Do not replace required functionality with placeholders.

Do not write:

`TODO`

for essential features and then declare the project complete.

---

# 56. DO NOT FABRICATE RESULTS

This is extremely important.

Do not put made-up:

- success rates;
- collision rates;
- training curves;
- planner timings;
- pose errors

into CSV files, figures or the report.

All numerical experimental results must come from actual program executions.

If training has not converged, report that honestly.

---

# 57. PROJECT SCOPE PRIORITY

If project complexity becomes excessive, prioritize in this order:

1. correct simulation;
2. correct joint-space action;
3. correct collision detection;
4. working RRT-Connect baseline;
5. working PPO;
6. noise-free/noisy comparison;
7. reproducible evaluation;
8. visual demo;
9. plots and report;
10. larger panel or extra experiments.

Do NOT sacrifice the core project in order to implement optional features.

---

# 58. FINAL PROJECT RESEARCH QUESTION

Use the following research question as the central framing:

**Can a PPO-based joint-space policy learn collision-aware robotic panel manipulation while maintaining accurate target alignment, and how robust is the learned policy to observation noise compared with a conventional sampling-based motion-planning and feedback-control pipeline?**

---

# 59. EXPECTED METHOD DIAGRAM

The implemented system should conceptually correspond to:

```text
                         TARGET PANEL POSE
                                |
                  +-------------+-------------+
                  |                           |
             CLASSICAL                       PPO
                  |                           |
                 IK                     Observation
                  |                           |
             q_goal                      PPO MLP
                  |                           |
           RRT-Connect                    Delta q
                  |                           |
       collision-free q path                  |
                  |                           |
                  +-------------+-------------+
                                |
                    Joint Position Controller
                                |
                              Robot
                                |
                           Panel motion
                                |
                  Collision / Pose Evaluation
```

---

# 60. FINAL OUTPUT EXPECTED FROM YOU

After implementation, provide a final development summary containing:

1. what was implemented;
2. repository file structure;
3. simulator and robot used;
4. final observation space;
5. final action space;
6. reward equation;
7. PPO hyperparameters;
8. RRT parameters;
9. noise definitions;
10. exact training commands;
11. exact evaluation commands;
12. exact demo commands;
13. test results;
14. generated model paths;
15. generated CSV paths;
16. generated figure paths;
17. any unresolved limitations;
18. any differences from this specification and why.

Before declaring completion, run the smoke test, tests, at least one GUI demo where possible, and representative evaluation commands.

The final repository must be understandable and reproducible by a course instructor without requiring knowledge from this conversation.