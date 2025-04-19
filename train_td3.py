import numpy as np
import matplotlib.pyplot as plt
from exploration_env_continuous import GridWorldEnv
from agent_td3 import TD3Agent

env = GridWorldEnv()
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]

agent = TD3Agent(obs_dim, act_dim)

episodes = 500
rewards = []
coverages = []
steps_list = []

def moving_avg(x, window=10):
    return np.convolve(x, np.ones(window) / window, mode='valid')

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    # Normalize state to [-1, 1]
    state = state.astype(np.float32) * 2 - 1
    noise_scale = max(0.01, 0.1 * (1 - ep / episodes))  # Linearly decay noise

    while not done:
        action = agent.act(state, noise_scale=noise_scale)
        action = np.clip(action, -1.0, 1.0)

        next_state, reward, done, _ = env.step(action)

        # Reward shaping
        y, x = env.agent_pos
        if env.map[y, x] == 0:
            reward = -2
        elif env.explored[y, x] == 1:
            reward = -0.1
        else:
            reward = 5

        if np.array_equal(state, next_state):
            reward -= 1

        coverage_ratio = np.sum(env.explored[env.map == 1]) / np.sum(env.map == 1)
        reward += 10 * coverage_ratio

        next_state = next_state.astype(np.float32) * 2 - 1
        agent.store(state, action, reward, next_state, float(done))

        if ep > 10:
            agent.update()

        state = next_state
        total_reward += reward
        steps += 1

    explored = np.sum(env.explored[env.map == 1])
    total = np.sum(env.map == 1)
    coverage = 100 * explored / total if total > 0 else 0

    rewards.append(total_reward)
    coverages.append(coverage)
    steps_list.append(steps)

    print(f"TD3 Episode {ep+1}: Reward = {total_reward:.1f}, Coverage = {coverage:.2f}%, Steps = {steps}, Noise = {noise_scale:.3f}")

# Plot smoothed reward
plt.plot(moving_avg(rewards))
plt.title("TD3: Smoothed Total Reward")
plt.xlabel("Episode")
plt.grid(True)
plt.show()

# Plot coverage
plt.plot(coverages)
plt.title("TD3: Coverage %")
plt.xlabel("Episode")
plt.grid(True)
plt.show()
