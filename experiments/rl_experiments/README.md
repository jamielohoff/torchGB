# CleanRL (Clean Implementation of RL Algorithms)

We base our RL experiments off of [CleanRL](https://github.com/vwxyzjn/cleanrl/tree/b09dec92646e973c8eccf9ffc92675e8ac72312c)

We reproduce the relevant ReadMe docs below.


CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. The implementation is clean and simple, yet we can scale it to run thousands of experiments using AWS Batch. 

You can read more about CleanRL in their [JMLR paper](https://www.jmlr.org/papers/volume23/21-1342/21-1342.pdf) and [documentation](https://docs.cleanrl.dev/).


> ℹ️ **Support for Gymnasium**: [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) is the next generation of [`openai/gym`](https://github.com/openai/gym) that will continue to be maintained and introduce new features. Please see their [announcement](https://farama.org/Announcing-The-Farama-Foundation) for further detail. We are migrating to `gymnasium` and the progress can be tracked in [vwxyzjn/cleanrl#277](https://github.com/vwxyzjn/cleanrl/pull/277). 

## Get started

Prerequisites:
* Python >=3.7.1,<3.11

To run experiments locally, give the following a try:

```bash
cd cleanrl

# `python run cleanrl/ppo.py`
python cleanrl/ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000

# open another terminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
```

To use experiment tracking with wandb, run
```bash
wandb login # only required for the first time
python cleanrl/ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000 \
    --track \
    --wandb-project-name cleanrltest
```

You can install CleanRL with `requirements.txt`:

```bash
# core dependencies
pip install -r requirements/requirements.txt

# optional dependencies
pip install -r requirements/requirements-atari.txt
pip install -r requirements/requirements-mujoco.txt
pip install -r requirements/requirements-mujoco_py.txt
pip install -r requirements/requirements-procgen.txt
pip install -r requirements/requirements-envpool.txt
pip install -r requirements/requirements-pettingzoo.txt
pip install -r requirements/requirements-jax.txt
pip install -r requirements/requirements-docs.txt
pip install -r requirements/requirements-cloud.txt
pip install -r requirements/requirements-memory_gym.txt
```

To run training scripts in other games:
```
# classic control
python cleanrl/ppo.py --env-id CartPole-v1

# atari
poetry install -E atari
python cleanrl/ppo_atari.py --env-id BreakoutNoFrameskip-v4

# procgen
poetry install -E procgen
python cleanrl/ppo_procgen.py --env-id starpilot

# ppo + lstm
poetry install -E atari
python cleanrl/ppo_atari_lstm.py --env-id BreakoutNoFrameskip-v4
```

## Algorithms Implemented


| Algorithm      | Variants Implemented |
| ----------- | ----------- |
| ✅ [Proximal Policy Gradient (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  |  [`ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py),   [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy) |
| |  [`ppo_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py),   [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy)
| |  [`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py),   [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)
| |  [`ppo_atari_lstm.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py),   [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy)
| |  [`ppo_atari_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py),   [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy)
| | [`ppo_atari_envpool_xla_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py), [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy)
| | [`ppo_atari_envpool_xla_jax_scan.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py), [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jax_scanpy))
| |  [`ppo_procgen.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py),   [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_procgenpy)
| |  [`ppo_atari_multigpu.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_multigpu.py),  [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_multigpupy)
| | [`ppo_pettingzoo_ma_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py),  [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy)
| | [`ppo_continuous_action_isaacgym.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py),  [docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy)
| | [`ppo_trxl.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_trxl/ppo_trxl.py),  [docs](https://docs.cleanrl.dev/rl-algorithms/ppo-trxl/) |

## Citing CleanRL

If you use CleanRL in your work, please cite our technical [paper](https://www.jmlr.org/papers/v23/21-1342.html):

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```


## Acknowledgement

CleanRL is a community-powered by project and our contributors run experiments on a variety of hardware.

* We thank many contributors for using their own computers to run experiments
* We thank Google's [TPU research cloud](https://sites.research.google/trc/about/) for providing TPU resources.
* We thank [Hugging Face](https://huggingface.co/)'s cluster for providing GPU resources. 
