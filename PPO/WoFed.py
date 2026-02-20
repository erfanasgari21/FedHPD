

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import gymnasium as gym

from collections import deque
from gymnasium.wrappers import RecordVideo

from rollout_buffer import RolloutBuffer

from src.utils.evaluation import record_episode


class Agent:
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=4,
        batch_size=64,
        device="cpu",
        action_low=None,
        action_high=None
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

        self.action_low = action_low
        self.action_high = action_high

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            action, log_prob, value = self.model.act(state)

            action = action.cpu().numpy()

            return action, log_prob.item(), value.item(), action


    def update(self):
        states = torch.tensor(self.buffer.states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(
            np.array(self.buffer.actions),
            dtype=torch.float32
        ).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs).to(self.device)

        returns, advantages = self.buffer.compute_returns_and_advantages(
            self.gamma, self.gae_lambda
        )
        returns = returns.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        loss_info = {"policy": [], "value": [], "entropy": []}

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                log_probs, entropy, values = self.model.evaluate(
                    states[batch_idx], actions[batch_idx].long()
                )

                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * advantages[batch_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[batch_idx] - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                loss_info["policy"].append(policy_loss.item())
                loss_info["value"].append(value_loss.item())
                loss_info["entropy"].append(entropy_loss.item())

        self.buffer.clear()
        return loss_info


class ScratchLogger:
    def __init__(self):
        self.rewards = []
        self.terminal_steps = []
        self.policy_losses = []
        self.value_losses = []
        self.loss_steps = []


def train_ppo(
    env,
    agent,
    run_id,
    total_timesteps=600_000,
    rollout_length=2048,
    log_interval=100,
    record_every=100_000,
    env_id="LunarLander-v3",
    continuous=False,
    video_folder="videos",
):

    logger = ScratchLogger()

    state, _ = env.reset()
    ep_reward = 0
    episode = 0

    reward_window = deque(maxlen=100)

    os.makedirs(video_folder, exist_ok=True)

    next_record_step = record_every

    for t in range(1, total_timesteps + 1):

        # -------- ACTION --------
        action, log_prob, value, stored_action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, stored_action, log_prob, reward, done, value)

        state = next_state
        ep_reward += reward

        # -------- EPISODE END --------
        if done:
            episode += 1

            reward_window.append(ep_reward)

            logger.rewards.append(ep_reward)
            logger.terminal_steps.append(t)

            if episode % log_interval == 0:
                print(
                    f"Episode {episode}, "
                    f"Timestep {t}, "
                    f"Mean(100) Reward: {np.mean(reward_window):.2f}"
                )

            state, _ = env.reset()
            ep_reward = 0

        # -------- PPO UPDATE --------
        if t % rollout_length == 0:
            loss_info = agent.update()

            logger.policy_losses.append(np.mean(loss_info["policy"]))
            logger.value_losses.append(np.mean(loss_info["value"]))
            logger.loss_steps.append(t)

        # -------- VIDEO RECORD (TIMESTEP BASED) --------
        if t >= next_record_step:
            print(f"Recording evaluation at timestep {t}")

            eval_env = gym.make(
                env_id,
                continuous=continuous,
                render_mode="rgb_array"
            )

            eval_env = RecordVideo(
                eval_env,
                video_folder=video_folder,
                episode_trigger=lambda x: True,
                name_prefix=f"{run_id}_step{t}"
            )

            record_episode(agent, eval_env)
            eval_env.close()

            next_record_step += record_every

    return logger


if __name__ == "__main__":

    set_seed(42)

    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        enable_wind=False,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DiscreteActorCritic(
        state_dim,
        action_dim
    )

    agent = PPOAgent(
        model,
        device="cpu",
    )

    logger = train_ppo(
        env,
        agent,
        run_id="scratch_discrete",
        env_id="LunarLander-v3",
        continuous=False,
        total_timesteps=600_000,
        rollout_length=2048,
    )

    plot_results(logger, title="From Scratch PPO - Discrete - Wind disabled")

    env.close()
