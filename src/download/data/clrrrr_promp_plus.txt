# Usage:

## Train:
e.g.
python run_scripts/pro-mp_run_mujoco.py —exp AntRandGoal --rollouts_per_meta_task 5

Envs:

AntRandGoal

HalfCheetahRandVel

HumanoidRandDirec2D

WalkerRandParams

## Test:
e.g.
python run_scripts/pro-mp_run_mujoco_test.py --dir run_1566926648 --eff 10  --exp AntRandGoal