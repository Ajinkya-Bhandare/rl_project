import matplotlib.pyplot as plt
import numpy as np
import torch
from exploration_env import GridWorldEnv
from ppo_agent import PPOAgent

env = GridWorldEnv()
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = PPOAgent(obs_shape, n_actions)

num_episodes = 500
all_rewards = []
all_coverages = []

for episode in range(num_episodes):
    state = env.reset()
    done = False

    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    total_reward = 0

    while not done:
        action, log_prob, _ = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        #env.render()  # Show map

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)

        total_reward += reward
        state = next_state

    # Estimate returns and advantages
    _, last_value = agent.model(torch.tensor(state).unsqueeze(0).float())
    returns = agent.compute_returns(rewards, dones, last_value.item())
    values = [agent.model(torch.tensor(s).unsqueeze(0).float())[1].item() for s in states]
    advantages = np.array(returns) - np.array(values)

    agent.update(states, actions, log_probs, returns, advantages)
    all_rewards.append(total_reward)

    # Exploration stats
    explored_count = np.sum(env.explored[env.map == 1])
    total_free = np.sum(env.map == 1)
    explored_percent = 100 * explored_count / total_free if total_free > 0 else 0
    all_coverages.append(explored_percent)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Coverage = {explored_percent:.2f}%", flush=True)

# Plot reward curve
plt.figure()
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Training on GridWorld")
plt.grid(True)
plt.show()

# Plot coverage curve
plt.figure()
plt.plot(all_coverages)
plt.xlabel("Episode")
plt.ylabel("Explored %")
plt.title("Exploration Coverage Over Time")
plt.grid(True)
plt.show()