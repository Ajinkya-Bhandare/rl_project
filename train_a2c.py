import torch
import numpy as np
import matplotlib.pyplot as plt
from exploration_env import GridWorldEnv
from agent_a2c import A2CAgent

env = GridWorldEnv()
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = A2CAgent(obs_shape, n_actions)

num_episodes = 200
all_rewards = []
all_coverages = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    states, actions, rewards, dones, log_probs = [], [], [], [], []
    total_reward = 0

    while not done:
        action, log_prob = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)

        state = next_state
        total_reward += reward

    agent.update(states, actions, rewards, dones, log_probs, next_state)

    # Track reward + coverage
    explored = np.sum(env.explored[env.map == 1])
    total = np.sum(env.map == 1)
    coverage = 100 * explored / total if total > 0 else 0
    all_rewards.append(total_reward)
    all_coverages.append(coverage)

    print(f"Episode {episode+1}: Reward = {total_reward}, Coverage = {coverage:.2f}%", flush=True)

plt.plot(all_rewards)
plt.title("A2C: Total Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

plt.plot(all_coverages)
plt.title("A2C: Map Coverage")
plt.xlabel("Episode")
plt.ylabel("Coverage %")
plt.grid(True)
plt.show()
