import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim[0] * obs_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim[0] * obs_dim[1] + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, a):
        x = torch.flatten(x, start_dim=1)
        return self.net(torch.cat([x, a], dim=1))

class TD3Agent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64, lr=1e-3, policy_delay=2, noise_std=0.2, noise_clip=0.5):
        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim, act_dim)
        self.critic2 = Critic(obs_dim, act_dim)
        self.critic1_target = Critic(obs_dim, act_dim)
        self.critic2_target = Critic(obs_dim, act_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.buffer = deque(maxlen=buffer_size)
        self.replay_buffer = self.buffer  # exposed for training script

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.total_it = 0
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.AdamW(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.AdamW(self.critic2.parameters(), lr=lr)

    def act(self, state, noise_scale=0.1):
        state = torch.tensor(state).unsqueeze(0).float()
        action = self.actor(state).detach().numpy()[0]
        noise = np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0)

    def store(self, *transition):
        self.buffer.append(transition)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return map(np.array, zip(*batch))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.sample()
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float().unsqueeze(1)

        # Optional reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            q1 = self.critic1_target(next_states, next_actions)
            q2 = self.critic2_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1, q2)

        q1_val = self.critic1(states, actions)
        q2_val = self.critic2(states, actions)
        critic1_loss = nn.functional.mse_loss(q1_val, q_target)
        critic2_loss = nn.functional.mse_loss(q2_val, q_target)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_opt.step()

        # Delay actor updates
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
            for target, source in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
            for target, source in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
