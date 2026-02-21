from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import threading
import copy
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from torch.distributions import Categorical
from policy import PPOActorCriticNetwork
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from rollout_buffer import RolloutBuffer

device = torch.device("cpu")

# PPO Hyperparameters
PPO_EPOCHS = 4          # Number of PPO update epochs per episode
PPO_CLIP = 0.2          # Clipping epsilon
VALUE_COEF = 0.5        # Coefficient for value loss
ENTROPY_COEF = 0.01     # Coefficient for entropy bonus
MAX_GRAD_NORM = 0.5     # Gradient clipping


def federated_average(probs):
    avg_probs = None
    for prob in probs:
        if avg_probs is None:
            avg_probs = prob.clone().detach()
        else:
            avg_probs += prob.clone().detach()
    return avg_probs / len(probs)


class Agent:
    def __init__(self, env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, device):
        self.env = env
        self.gamma = gamma
        self.device = device
        # Use PPO Actor-Critic network instead of plain policy network
        self.policy = PPOActorCriticNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.buffer = RolloutBuffer()

        # PPO minibatch settings (مثل non-fed)
        self.gae_lambda = 0.95
        self.epochs     = PPO_EPOCHS
        self.batch_size = 64
        self.kd_optimizer = optim.SGD(self.policy.actor_head.parameters(), lr=lr * 0.5)
        
    def collect_rollout(self, rollout_length):
      # شروع: اگر state نگه نمی‌داری، اینجا reset کن
      obs, _ = self.env.reset()
      self.buffer.clear()
      episodic_returns = []
      ep_return = 0.0

      for t in range(rollout_length):
          state = torch.tensor(obs, dtype=torch.float32, device=self.device)

          with torch.no_grad():
              action, log_prob, value = self.policy.act(state)

          next_obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
          raw_reward = reward               # ✅ برای لاگ اپیزودی
          ep_return += float(raw_reward)    # ✅ episodic return واقعی
          # ✅ TimeLimit bootstrap (اگر truncation بود ولی termination نبود)
          if truncated and (not terminated):
              with torch.no_grad():
                  s_next = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                  _, v_next = self.policy.forward(s_next)
                  reward = reward + self.gamma * float(v_next.item())

          done = terminated or truncated
          if done:
            episodic_returns.append(ep_return)
            ep_return = 0.0

          self.buffer.add(obs, int(action.item()), float(log_prob.item()), reward, done, float(value.item()))
          obs = next_obs

          if done:
              obs, _ = self.env.reset()

      # bootstrap value برای آخر rollout
      last_done = self.buffer.dones[-1]
      if last_done:
          last_value = 0.0
      else:
          with torch.no_grad():
              s = torch.tensor(obs, dtype=torch.float32, device=self.device)
              _, v = self.policy.forward(s)
              last_value = float(v.item())
      if len(episodic_returns) > 0:
          reported_reward = float(np.mean(episodic_returns))  # ✅ میانگین اپیزودهای کامل داخل rollout
      else:
          reported_reward = float(ep_return)  # اگر هیچ اپیزود کاملی تمام نشد

      return last_value, reported_reward
     
    def update_ppo(self, last_value):
      states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
      actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
      old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)

      returns, advantages = self.buffer.compute_returns_and_advantages(
          gamma=self.gamma, gae_lambda=self.gae_lambda, last_value=last_value
      )
      returns = returns.to(self.device)
      advantages = advantages.to(self.device)
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      dataset_size = states.size(0)
      indices = np.arange(dataset_size)
      policy_losses = []
      value_losses  = []
      entropies     = []
      approx_kls    = []
      clipfracs     = []
      for _ in range(self.epochs):
          np.random.shuffle(indices)
          for start in range(0, dataset_size, self.batch_size):
              end = start + self.batch_size
              batch_idx = indices[start:end]

              log_probs, entropy, values = self.policy.evaluate(
                  states[batch_idx], actions[batch_idx]
              )

              ratio = torch.exp(log_probs - old_log_probs[batch_idx])
              surr1 = ratio * advantages[batch_idx]
              surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages[batch_idx]

              actor_loss  = -torch.min(surr1, surr2).mean()
              critic_loss = VALUE_COEF * (returns[batch_idx] - values).pow(2).mean()
              entropy_loss = entropy.mean()
              with torch.no_grad():
                  # approx_kl ≈ mean(old_logp - new_logp)
                  approx_kl = (old_log_probs[batch_idx] - log_probs).mean().item()

                  # clipfrac: درصد ratioهایی که کلیپ شده‌اند
                  clipfrac = ((ratio - 1.0).abs() > PPO_CLIP).float().mean().item()

              policy_losses.append(actor_loss.item())
              value_losses.append(critic_loss.item())
              entropies.append(entropy_loss.item())
              approx_kls.append(approx_kl)
              clipfracs.append(clipfrac)

              loss = actor_loss + critic_loss - ENTROPY_COEF * entropy_loss

              self.optimizer.zero_grad()
              loss.backward()
              torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
              self.optimizer.step()

      self.scheduler.step()
      self.buffer.clear()
      return {
          "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
          "value_loss":  float(np.mean(value_losses))  if value_losses  else 0.0,
          "entropy":     float(np.mean(entropies))     if entropies     else 0.0,
          "approx_kl":   float(np.mean(approx_kls))    if approx_kls    else 0.0,
          "clipfrac":    float(np.mean(clipfracs))     if clipfracs     else 0.0,
      }
    def calculate_kd_loss(self, sampled_states_part, avg_probs):
      """
      KD فقط روی actor_head آپدیت می‌کند (نه shared و نه critic).
      """

      # ✅ shared بدون گرادیان (قطع مسیر backprop به بدنه مشترک)
      with torch.no_grad():
          x = self.policy.shared(sampled_states_part)

      # ✅ logits فقط از actor_head (گرادیان فقط برای actor_head ساخته می‌شود)
      logits = self.policy.actor_head(x)
      student = Categorical(logits=logits)

      with torch.no_grad():
          teacher = Categorical(probs=avg_probs)

      KD_ALPHA = 1
      kd_loss = KD_ALPHA * torch.distributions.kl.kl_divergence(student, teacher).mean()

      # ✅ فقط actor_head آپدیت می‌شود
      self.kd_optimizer.zero_grad()
      kd_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy.actor_head.parameters(), MAX_GRAD_NORM)
      self.kd_optimizer.step()

      return kd_loss.item()


class Server:
    def __init__(self, agents, device):
        self.agents = agents
        self.device = device

    def collect_and_average_probs(self, sampled_states):
        agent_probs = []
        with torch.no_grad():  # ✅ این اضافه شد
            for agent in self.agents:
                logits, _ = agent.policy(sampled_states)
                probs = torch.softmax(logits, dim=-1)
                agent_probs.append(probs)

            avg_probs = torch.mean(torch.stack(agent_probs, dim=0), dim=0)
            avg_probs = avg_probs / (avg_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # برای اطمینان از ثابت بودن و عدم ساخت گراف
        return [p.detach() for p in agent_probs], avg_probs.detach()


def main(seed, episodes, distill_interval, run):
    agents_count = 3
    seeds = [seed + 1000*i for i in range(agents_count)]

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
    agents = []
    for i in range(agents_count):
        np.random.seed(seeds[i])
        torch.manual_seed(seeds[i])   # مهم: قبل از ساخت شبکه/مدل

        agents.append(
            Agent(
                envs[i], state_size, action_size,
                hidden_sizes_list[i % len(hidden_sizes_list)],
                lr_list[i % len(lr_list)],
                activation_list[i % len(activation_list)],
                gamma, device=device
            )
        )

    server = Server(agents, device)

    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards   = []
    kd_losses         = {f'Agent {i + 1}': [] for i in range(agents_count)}

# ── Setup Live Logging ────────────────────────────────────────────────────
    timestamp_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ساخت فایل متنی برای لاگ زنده
    log_file = open(f"results/live_log_FedPPO_run{run}_{timestamp_start}.txt", "w")
    log_file.write("Episode,System_Avg_Reward,Rolling_100_Avg\n")
    
    # راه اندازی تنسوربورد
    writer = SummaryWriter(log_dir=f"runs/FedPPO_run{run}_{timestamp_start}")
    pbar = tqdm(range(episodes), unit="ep")
    for episode in pbar:

        # ── Parallel PPO training via ThreadPoolExecutor ──────────────────────
        rollout_length = 2048  # می‌تونی بیرون حلقه هم تعریفش کنی

        with ThreadPoolExecutor(max_workers=agents_count) as executor:
            rollout_results = list(executor.map(lambda a: a.collect_rollout(rollout_length), agents))

        last_values   = [x[0] for x in rollout_results]
        total_rewards = [x[1] for x in rollout_results]
        ppo_infos = []
        for agent, last_v in zip(agents, last_values):
            info = agent.update_ppo(last_v)
            ppo_infos.append(info)
        for i, reward in enumerate(total_rewards):
            rewards_per_agent[i].append(reward)

        average_reward = sum(total_rewards) / len(total_rewards)
        average_rewards.append(average_reward)
        # تعریف محاسبه timestep دقیقاً مشابه کد بیس‌لاین
        timestep = (episode + 1) * rollout_length

        # ✅ Rolling-100 برای هر ایجنت (برای نمودار کم‌نوسان‌تر)
        for i in range(agents_count):
            window = rewards_per_agent[i][-100:]
            agent_roll100 = sum(window) / len(window)
            writer.add_scalar(f"Agents_Rolling100/Agent_{i+1}", agent_roll100, timestep)

        # Compute rolling 100-episode average across all agents
        all_recent = [
            rewards_per_agent[i][-100:] for i in range(agents_count)
        ]
        rolling_avg = sum(sum(r) / len(r) for r in all_recent) / agents_count

        pbar.set_postfix({
            "avg_reward": f"{average_reward:.2f}",
            "rolling_100": f"{rolling_avg:.2f}"
        })
        # ── Write Live Logs ───────────────────────────────────────────────────
        # ذخیره در فایل متنی و اجبار سیستم به نوشتن آنی (flush)
        log_file.write(f"{episode + 1},{average_reward:.2f},{rolling_avg:.2f}\n")
        log_file.flush()
        
        # ارسال اطلاعات به تنسوربورد
        writer.add_scalar("Reward/System_Episode_Avg", average_reward, timestep)
        writer.add_scalar("Reward/System_Rolling_100", rolling_avg, timestep)
        # ✅ PPO metrics (system average)
        writer.add_scalar("PPO/policy_loss", np.mean([x["policy_loss"] for x in ppo_infos]), timestep)
        writer.add_scalar("PPO/value_loss",  np.mean([x["value_loss"]  for x in ppo_infos]), timestep)
        writer.add_scalar("PPO/entropy",     np.mean([x["entropy"]     for x in ppo_infos]), timestep)
        writer.add_scalar("PPO/approx_kl",   np.mean([x["approx_kl"]   for x in ppo_infos]), timestep)
        writer.add_scalar("PPO/clipfrac",    np.mean([x["clipfrac"]    for x in ppo_infos]), timestep)
        # ارسال پاداش هر ایجنت به صورت جداگانه
        for idx, reward in enumerate(total_rewards):
            writer.add_scalar(f"Agents_Reward/Agent_{idx + 1}", reward, timestep)
        # print(f"Episode {episode + 1}:")
        # for idx, reward in enumerate(total_rewards):
        #     history = rewards_per_agent[idx]
        #     window   = history[-100:] if len(history) >= 100 else history
        #     last_avg = sum(window) / len(window)
        #     print(f"  Agent {idx + 1} Reward: {reward:.2f} | Avg(100): {last_avg:.4f}")
        # print(f"  Average Reward: {average_reward:.4f}")



        # ── Federated Knowledge Distillation ──────────────────────────────────
        if (episode + 1) % distill_interval == 0:
          csv_file = 'KD_state_10-from-each.csv'
          df = pd.read_csv(csv_file)
          sampled_states = df['State'].apply(
              lambda x: torch.tensor([float(v) for v in x.split(',')])
          )
          sampled_states = torch.stack(sampled_states.values.tolist()).to(device)

          dataset_size = sampled_states.size(0)
          assert dataset_size == 10000, "sampled_states size should be 10000"

          quarter_size = dataset_size // 4
          sampled_states_parts = [
              sampled_states[i * quarter_size: (i + 1) * quarter_size]
              for i in range(4)
          ]

          # ✅ 1) teacher را برای هر part فقط یک‌بار و قبل از هر آپدیت KD بساز
          avg_probs_parts = []
          for part in sampled_states_parts:
              _, avg_probs = server.collect_and_average_probs(part)   # no_grad داخل سرور هست
              avg_probs_parts.append(avg_probs)                       # detach شده

          distill_loop = 5
          for distillation_round in range(distill_loop):
              for part_idx, sampled_states_part in enumerate(sampled_states_parts):
                  avg_probs_fixed = avg_probs_parts[part_idx]  # ✅ teacher ثابت

                  for agent_idx, agent in enumerate(agents):
                      kd_loss = agent.calculate_kd_loss(sampled_states_part, avg_probs_fixed)

                      kd_losses[f'Agent {agent_idx + 1}'].append(
                          (episode + 1, distillation_round + 1, kd_loss)
                      )
                      writer.add_scalar(f"KD/loss_agent_{agent_idx+1}", kd_loss, timestep)

              tqdm.write(f"Episode {episode + 1}: KD distillation completed.")
    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rewards_df = pd.DataFrame({f'Agent {i + 1}': rewards_per_agent[i] for i in range(agents_count)})
    rewards_df.to_csv(f'results/rewards_per_agent_Lunar_PPO_{run}_{timestamp}.csv', index=False)

    avg_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Average Reward': average_rewards})
    avg_df.to_csv(f'results/rewards_Lunar_PPO_{run}_{timestamp}.csv', index=False)
    log_file.close()
    writer.close()


if __name__ == "__main__":
    for run in range(1):
        print(f"\nRun {run + 1}:")
        seed = 20 + run * 5
        episodes = 500
        distill_interval = 5
        print(f"Seed: {seed}, Episodes: {episodes}, Distill Interval: {distill_interval}")
        main(seed, episodes, distill_interval, run)