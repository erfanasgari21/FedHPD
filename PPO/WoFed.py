from torch.utils.tensorboard import SummaryWriter
from rollout_buffer import RolloutBuffer
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import gymnasium as gym
from tqdm import tqdm
from datetime import datetime
import pandas as pd

from collections import deque
from gymnasium.wrappers import RecordVideo

from policy import PPOActorCriticNetwork
from rollout_buffer import RolloutBuffer
from concurrent.futures import ThreadPoolExecutor

class Agent:
    def __init__(
        self,
        env,
        model,
        lr=3e-4,
        gamma=0.99,
        device="cpu",
        
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=4,
        batch_size=64,
    ):

        self.env = env

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

        self.state, _ = env.reset()
        self.ep_reward = 0.0
        self.rewards = []
        self.terminal_steps = []

    def collect_rollout(self, T, rollout_length):
        """Collect exactly rollout_length steps, spanning episode boundaries."""
        for t in range(rollout_length):
            action, log_prob, value = self.select_action(self.state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(self.state, action, log_prob, reward, done, value)
            self.state     = next_state
            self.ep_reward += reward

            if done:
                self.rewards.append(self.ep_reward)
                self.terminal_steps.append(T*rollout_length + t)
                self.ep_reward = 0.0
                self.state, _ = self.env.reset()


    def select_action(self, state):
      state = torch.tensor(state, dtype=torch.float32, device=self.device)

      with torch.no_grad():
          action, log_prob, value = self.model.act(state)

      action = int(action.cpu().item())  # ✅ اکشن گسسته باید int باشد
      return action, float(log_prob.item()), float(value.item())


    def update(self):
        states        = torch.tensor(np.array(self.buffer.states),  dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs,         dtype=torch.float32).to(self.device)

        # ---- 1) محاسبه last_value برای bootstrap ----
        last_done = self.buffer.dones[-1]
        if last_done:
            last_value = 0.0
        else:
            with torch.no_grad():
                s = torch.tensor(self.state, dtype=torch.float32, device=self.device)
                _, v = self.model.forward(s)
                last_value = v.item()

        # ---- 2) حالا GAE درست محاسبه میشه ----
        returns, advantages = self.buffer.compute_returns_and_advantages(
            self.gamma, self.gae_lambda, last_value
        )
        returns = returns.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        # loss_info = {"policy": [], "value": [], "entropy": []}

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
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[batch_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[batch_idx] - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # loss_info["policy"].append(policy_loss.item())
                # loss_info["value"].append(value_loss.item())
                # loss_info["entropy"].append(entropy_loss.item())

        self.buffer.clear()
        # return loss_info



# def ppo_train(
#     env,
#     agent,
#     run_id,
#     total_timesteps=600_000,
#     rollout_length=1024,
#     log_interval=100,
#     record_every=100_000,
#     env_id="LunarLander-v3",
#     continuous=False,
#     video_folder="videos",
# ):

#     logger = ScratchLogger()

#     ep_reward = 0
#     episode = 0

#     reward_window = deque(maxlen=100)


#     for t in range(1, total_timesteps + 1):

#         # -------- ACTION --------
#         action, log_prob, value, stored_action = agent.select_action(state)

#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#         agent.buffer.add(state, stored_action, log_prob, reward, done, value)

#         state = next_state
#         ep_reward += reward

#         # -------- EPISODE END --------
#         if done:
#             episode += 1

#             reward_window.append(ep_reward)

#             logger.rewards.append(ep_reward)
#             logger.terminal_steps.append(t)

#             if episode % log_interval == 0:
#                 print(
#                     f"Episode {episode}, "
#                     f"Timestep {t}, "
#                     f"Mean(100) Reward: {np.mean(reward_window):.2f}"
#                 )

#             state, _ = env.reset()
#             ep_reward = 0

#         # -------- PPO UPDATE --------
#         if t % rollout_length == 0:
#             loss_info = agent.update()

#             logger.policy_losses.append(np.mean(loss_info["policy"]))
#             logger.value_losses.append(np.mean(loss_info["value"]))
#             logger.loss_steps.append(t)


#     return logger


def main(seed, num_updates, run):
    agents_count = 10
    seeds = [seed + i*1000 for i in range(agents_count)]

    # فقط یک بار seed کلی
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ساخت envها با seed جدا
    envs = [gym.make('LunarLander-v3') for _ in range(agents_count)]
    for i, env in enumerate(envs):
        env.reset(seed=seeds[i])

    state_size  = envs[0].observation_space.shape[0]
    action_size = envs[0].action_space.n

    lr_list = [1e-4, 1e-4, 1e-4, 1e-4, 3e-4,
               4e-4, 5e-4, 1e-4, 2e-4, 1e-4]
    activation_list = ['relu', 'relu', 'tanh', 'relu', 'tanh',
                       'relu', 'tanh', 'relu', 'tanh', 'relu']
    gamma = 0.99
    rollout_length = 2048
    hidden_sizes_list = [[128, 128, 256], [64, 64], [128, 128], [128, 256], [256, 256],
                         [512], [64, 128, 64], [32, 32], [512, 512], [1024]]

    # ساخت Agentها (هر کدام مدل با seed خودش)
    agents = []
    for i in range(agents_count):
        torch.manual_seed(seeds[i])  # ✅ وزن اولیه هر مدل متفاوت و reproducible

        model = PPOActorCriticNetwork(
            state_size, action_size,
            hidden_sizes_list[i % len(hidden_sizes_list)],
            activation_list[i % len(activation_list)]
        )

        agents.append(
            Agent(
                envs[i],
                model,
                lr=lr_list[i % len(lr_list)],
                gamma=gamma,
                entropy_coef=0.01,
            )
        )

    all_rewards = [[] for _ in range(agents_count)] 
    all_terminal_steps = [[] for _ in range(agents_count)] 

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = open(f"training_log_WoFed_{timestamp}.txt", "w")
    log_file.write("Update,Timestep,Rolling_Avg_Reward\n") # نوشتن هدر

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"runs/WoFed_{timestamp}")
    
    pbar = tqdm(range(num_updates), unit="update")
    for T in pbar:
        
        # Collect rollouts in parallel
        with ThreadPoolExecutor(max_workers=agents_count) as executor:
            executor.map(lambda a: a.collect_rollout(T, rollout_length), agents)
        
        # Update all agents
        for agent in agents:
            agent.update()

        # Gather any completed episodes this rollout
        for i, agent in enumerate(agents):
            all_rewards[i].extend(agent.rewards)
            all_terminal_steps[i].extend(agent.terminal_steps)
            agent.rewards, agent.terminal_steps = [], []
        
        rolling_avgs = []
        for i in range(agents_count):
            window = all_rewards[i][-100:] if all_rewards[i] else [0]
            rolling_avgs.append(sum(window) / len(window))

        avg_rolling = sum(rolling_avgs) / agents_count
        timestep    = (T + 1) * rollout_length

        pbar.set_postfix({
            "timestep":    timestep,
            "rolling_100": f"{avg_rolling:.2f}"
        })
        
        
        # --- اضافه کردن این دو خط برای ذخیره آنی لاگ ---
        log_file.write(f"{T},{timestep},{avg_rolling:.2f}\n")
        log_file.flush()  # این دستور باعث می‌شود اطلاعات فوراً در فایل ذخیره شود

        # ارسال میانگین پاداش کل سیستم به تنسوربورد
        writer.add_scalar("Reward/System_Average", avg_rolling, timestep)
        
        # ارسال پاداش هر ایجنت به صورت جداگانه
        for i in range(agents_count):
            writer.add_scalar(f"Reward/Agent_{i+1}", rolling_avgs[i], timestep)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Pad to same length for DataFrame
    max_len    = max(len(r) for r in all_rewards)
    rewards_df = pd.DataFrame({
        f'Agent {i+1}': all_rewards[i] + [np.nan] * (max_len - len(all_rewards[i]))
        for i in range(agents_count)
    })
    log_file.close()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"runs/WoFed_{timestamp}")
    
    rewards_df.to_csv(f'rewards_per_agent_Lunar_PPO_NoFed_{run}_{timestamp}.csv', index=False)



if __name__ == "__main__":
    for run in range(1):
        print(f"\nRun {run + 1}:")
        seed  = 20 + run * 5
        num_updates=500
        print(f"Seed: {seed}")
        main(seed, num_updates, run)

