import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(ActorCritic, self).__init__()
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
        x = x.unsqueeze(1).float()  # Add channel dimension
        conv_out = self.conv(x)
        return self.actor(conv_out), self.critic(conv_out)

class PPOAgent:
    def __init__(self, obs_shape, n_actions, lr=2.5e-4, gamma=0.99, clip_eps=0.2):
        self.model = ActorCritic(obs_shape, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def get_action(self, state):
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(state).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), dist.entropy()

    def compute_returns(self, rewards, dones, last_value):
        R = last_value
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions)
        log_probs_old = torch.stack(log_probs_old).detach()
        returns = torch.tensor(returns).float().detach()
        advantages = torch.tensor(advantages).float().detach()

        for _ in range(4):  # Multiple epochs
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
