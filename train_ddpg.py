import numpy as np
import matplotlib.pyplot as plt
from exploration_env_continuous import GridWorldEnv
from agent_ddpg import DDPGAgent

env = GridWorldEnv()
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]

agent = DDPGAgent(obs_dim, act_dim)

episodes = 200
rewards = []
coverages = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, float(done))
        agent.update()
        state = next_state
        total_reward += reward

    explored = np.sum(env.explored[env.map == 1])
    total = np.sum(env.map == 1)
    coverage = 100 * explored / total if total > 0 else 0
    rewards.append(total_reward)
    coverages.append(coverage)
    print(f"Episode {ep+1}: Reward = {total_reward:.1f}, Coverage = {coverage:.2f}%")

plt.plot(rewards)
plt.title("DDPG: Total Reward")
plt.grid(True)
plt.show()

plt.plot(coverages)
plt.title("DDPG: Coverage %")
plt.grid(True)
plt.show()
