import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Import Genomic Bottleneck
from torchGB import GenomicBottleneck


@dataclass
class Args:
    exp_name: str = "ppo_gnets"
    """The name of this experiment."""

    seed: int = 1
    """Seed for reproducibility."""

    torch_deterministic: bool = True
    """If True, ensures deterministic computation in torch."""

    cuda: bool = True
    """If True, enables CUDA for training."""

    track: bool = False
    """If True, logs results using Weights and Biases (wandb)."""

    wandb_project_name: str = "cleanRL"
    """The wandb project name."""

    wandb_entity: str = None
    """The entity (team) name in wandb."""

    capture_video: bool = False
    """Whether to capture videos of the agent's performance in the 'videos' folder."""

    env_id: str = "CartPole-v1"
    """The id of the environment to train in."""

    total_timesteps: int = 500000
    """Total number of training timesteps."""

    learning_rate: float = 2.5e-4
    """The learning rate of the optimizer."""

    num_envs: int = 4
    """The number of parallel environments."""

    num_steps: int = 128
    """The number of steps per rollout in each environment."""

    anneal_lr: bool = True
    """If True, linearly decays learning rate over training."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    gae_lambda: float = 0.95
    """Lambda factor for Generalized Advantage Estimation (GAE)."""

    num_minibatches: int = 4
    """Number of mini-batches used for PPO updates."""

    update_epochs: int = 4
    """Number of epochs for PPO policy updates per batch."""

    norm_adv: bool = True
    """If True, normalizes advantages."""

    clip_coef: float = 0.2
    """The PPO surrogate loss clipping coefficient."""

    clip_vloss: bool = True
    """If True, clips the value function loss."""

    ent_coef: float = 0.01
    """Entropy coefficient for the loss function."""

    vf_coef: float = 0.5
    """Value function coefficient for the loss function."""

    max_grad_norm: float = 0.5
    """Maximum gradient norm for gradient clipping."""

    target_kl: float = None
    """Target KL divergence threshold for early stopping."""

    checkpoint: bool = False
    """If True, saves model checkpoints after training."""

    gnet_enabled: bool = True
    """If True, enables Genomic Bottleneck (gnets) for weight compression."""

    gnet_ignore_layers: str = ""
    """Comma-separated list of layers to ignore when applying Genomic Bottleneck."""

    gnet_checkpoint: str = None
    """Path to load saved gnet weights, if any."""


def make_env(env_id, idx, capture_video, run_name):
    """Creates an environment instance with optional video recording."""
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes neural network layers with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent with separate actor and critic networks."""
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Setup environment and reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    # Initialize model
    agent = Agent(envs).to(device)

    # Wrap model with Genomic Bottleneck (if enabled)
    if args.gnet_enabled:
        num_batches = args.total_timesteps // (args.batch_size * args.num_minibatches)
        gnets = GenomicBottleneck(agent, num_batches)

        if args.gnet_checkpoint:
            gnets.load(args.gnet_checkpoint)
            gnets.predict_weights()

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Training loop
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            with torch.no_grad():
                if args.gnet_enabled:
                    gnets.predict_weights()  # Ensure updated model weights
                action, logprob, _, value = agent.get_action_and_value(torch.zeros(1).to(device))
        
        # PPO update
        optimizer.zero_grad()
        if args.gnet_enabled:
            gnets.zero_grad()
            gnets.predict_weights()
        loss = torch.tensor(0.0, requires_grad=True)  # Placeholder loss calculation
        loss.backward()
        if args.gnet_enabled:
            gnets.backward()
        optimizer.step()
        if args.gnet_enabled:
            gnets.step()

    # Save models
    if args.checkpoint:
        torch.save(agent.state_dict(), f"models/{run_name}_ppo_agent.pth")
        if args.gnet_enabled:
            gnets.save(f"models/{run_name}_gnets.pth")

    envs.close()