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

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64, lr=1e-3):
        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.buffer = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, state, noise_scale=0.1):
        state = torch.tensor(state).unsqueeze(0).float()
        action = self.actor(state).detach().numpy()[0]
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)

    def store(self, *transition):
        self.buffer.append(transition)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return map(np.array, zip(*batch))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample()
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float().unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q

        q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft updates
        for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
