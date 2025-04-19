import numpy as np
import matplotlib.pyplot as plt
from exploration_env_continuous import GridWorldEnv
from agent_ddpg import DDPGAgent

env = GridWorldEnv(max_steps=500)  # Updated to 500 steps
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]

agent = DDPGAgent(obs_dim, act_dim)

episodes = 500
rewards = []
coverages = []
steps_per_episode = []

def moving_avg(x, window=10):
    return np.convolve(x, np.ones(window) / window, mode='valid')

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    noise_scale = max(0.01, 0.1 * (1 - ep / episodes))

    while not done:
        state_norm = state.astype(np.float32) * 2 - 1
        action = agent.act(state_norm, noise_scale=noise_scale)
        action = np.clip(action, -1.0, 1.0)

        next_state, reward, done, _ = env.step(action)

        # Apply reward shaping after 20 episodes
        y, x = env.agent_pos
        if ep > 20:
            if env.map[y, x] == 0:
                reward = -5
            elif env.explored[y, x] == 1:
                reward = -0.2
            else:
                reward = 15

            # Penalize stagnation
            if np.array_equal(state, next_state):
                reward -= 1

        # Encourage coverage
        coverage_ratio = np.sum(env.explored[env.map == 1]) / np.sum(env.map == 1)
        reward += 10 * coverage_ratio

        next_state_norm = next_state.astype(np.float32) * 2 - 1
        agent.store(state_norm, action, reward, next_state_norm, float(done))

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
    steps_per_episode.append(steps)

    print(f"DDPG Episode {ep+1}: Reward = {total_reward:.1f}, Coverage = {coverage:.2f}%, Steps = {steps}, Noise = {noise_scale:.3f}")

# Plot reward
plt.plot(moving_avg(rewards))
plt.title("DDPG: Smoothed Total Reward")
plt.xlabel("Episode")
plt.grid(True)
plt.show()

# Plot coverage
plt.plot(coverages)
plt.title("DDPG: Coverage %")
plt.xlabel("Episode")
plt.grid(True)
plt.show()
