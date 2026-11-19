# On-path vertical weaving scene v25

The selected scene retains four differently sized, moving obstacles and the
current PPO plus predictive control plus local recovery. Robot start, target,
policy reference, success tolerances, collision rules and 240-step budget are
unchanged. No model training or video export was performed.

Physical boxes are checked against measured/nominal arm guides at their anchors.
O1 and O3 receive small anchor translations; O1 also receives a phase change.
The old execution would collide with O1 at simulation time 1.5 s. This timed
conflict is checked using real box geometry, not an enlarged swept box.

The selected new physical execution succeeds in 77 steps, with position error
0.332 mm and orientation error 0.0244 degrees. Measured whole-panel passages are
below O1, above O2, then below O4. Their heights are 0.514, 0.695 and 0.546 m.
This is three alternating passages, not four.

The initial full reanchor pilot timed out. Four resampled layouts then produced
two successes and two timeouts, with no qualifying alternating passages. A minimal
corridor refinement succeeded with three passages, followed by the selected
time-aligned variant. All outcomes remain in this folder. These development
trials selected a demonstration and do not estimate independent success rate.

Generator: `scripts/adjust_weaving_scene.py`
Parameters: `configs/on_path_weaving.yaml`
Selected scene/outcome/recording: `timed_crossing_demo/`
Portable replay: `demo/on_path_weaving_v25.json` in the v24 experiment folder.
