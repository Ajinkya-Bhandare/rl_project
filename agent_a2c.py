import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class A2CModel(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = 16 * obs_shape[0] * obs_shape[1]

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1).float()  # Add channel dim
        flat = self.conv(x)
        return self.actor(flat), self.critic(flat)

class A2CAgent:
    def __init__(self, obs_shape, n_actions, gamma=0.99, lr=1e-3):
        self.model = A2CModel(obs_shape, n_actions)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state):
        logits, _ = self.model(torch.tensor(state).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, rewards, dones, log_probs, next_state):
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        log_probs = torch.stack(log_probs)

        _, next_value = self.model(torch.tensor(next_state).unsqueeze(0).float())
        values = torch.cat([self.model(s.unsqueeze(0))[1] for s in states]).squeeze()
        returns = []
        R = next_value.item()

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns)

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
