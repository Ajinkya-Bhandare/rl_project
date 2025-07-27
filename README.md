# Reinforcement Learning for Efficient Exploration

This repository contains a comprehensive reinforcement learning research project focused on **efficient exploration** in grid-world environments. The project implements and compares multiple state-of-the-art RL algorithms including discrete methods (DQN, Actor-Critic) and continuous control methods (DDPG, TD3) for robotic exploration tasks. Our research demonstrates that continuous control provides smoother, more expressive navigation strategies suitable for real-world robotic applications.

## Project Overview

This research project explores discrete and continuous control methods for efficient robotic exploration in 2D grid-world environments. We implement and compare Deep Q-Networks (DQN), Actor-Critic (A2C), Deep Deterministic Policy Gradient (DDPG), and Twin Delayed DDPG (TD3) algorithms. Our goal is to maximize coverage of a 100×100 binary grid map generated using Perlin noise, where each cell represents free or occupied space.

**Key Research Findings:**
- Continuous control methods (DDPG, TD3) provide smoother, more expressive navigation strategies
- TD3 outperforms DDPG with faster convergence and lower variance in performance
- Reward shaping is crucial for efficient learning in sparse-reward exploration tasks
- Continuous methods achieve over 80% coverage in best cases

## Project Structure

```
rl_project/
├── agent_a2c.py              # Advantage Actor-Critic (A2C) agent implementation
├── agent_ddpg.py             # Deep Deterministic Policy Gradient (DDPG) agent
├── agent_td3.py              # Twin Delayed DDPG (TD3) agent
├── ppo_agent.py              # Proximal Policy Optimization (PPO) agent
├── exploration_env.py         # Discrete action space exploration environment
├── exploration_env_continuous.py  # Continuous action space exploration environment
├── train_a2c.py              # Training script for A2C
├── train_ddpg.py             # Training script for DDPG
├── train_td3.py              # Training script for TD3
├── train_ppo.py              # Training script for PPO
├── utils.py                  # Utility functions and helpers
├── flood_fill.py             # Flood fill algorithm implementation
├── perlin_noise.py           # Perlin noise generation for environment dynamics
├── numpy_test.py             # NumPy testing utilities
└── Exploration-grid-world-final.ipynb  # Jupyter notebook with analysis and results
```

## Algorithms Implemented

### 1. **Deep Q-Networks (DQN)**
- **File**: `agent_dqn.py`, `train_dqn.py`
- **Description**: Value-based method that approximates Q-value function using neural networks
- **Architecture**: 3 convolution layers followed by 3 fully connected layers
- **Features**: Experience replay buffer, target network updates every 1000 steps
- **Best for**: Discrete action spaces, stable Q-learning

### 2. **Actor-Critic (A2C)**
- **File**: `agent_a2c.py`, `train_a2c.py`
- **Description**: On-policy algorithm that combines policy gradient methods with value function approximation
- **Architecture**: 3 convolution layers followed by 3 fully connected layers for both actor and critic
- **Features**: Dual network structure (actor + critic), Adam optimizer with 5×10^-3 learning rate
- **Best for**: Discrete action spaces, stable learning

### 3. **Deep Deterministic Policy Gradient (DDPG)**
- **File**: `agent_ddpg.py`, `train_ddpg.py`
- **Description**: Off-policy actor-critic algorithm for continuous action spaces
- **Architecture**: Single actor-critic with continuous action space [-1,1]^2
- **Features**: 
  - Movement vector (dx,dy) output for smooth navigation
  - Target networks with soft updates
  - Gaussian noise for exploration with decaying scale
  - Replay buffer for off-policy learning
- **Best for**: Continuous control tasks, robotics applications

### 4. **Twin Delayed DDPG (TD3)**
- **File**: `agent_td3.py`, `train_td3.py`
- **Description**: Improved version of DDPG with twin critics and delayed policy updates
- **Architecture**: Dual critic networks with minimum Q-value selection
- **Features**: 
  - Twin critics to reduce overestimation bias
  - Delayed policy updates for stability
  - Target policy smoothing with clipped noise
  - Same continuous action space as DDPG
- **Best for**: Continuous control with improved stability and performance

### 5. **Proximal Policy Optimization (PPO)**
- **File**: `ppo_agent.py`, `train_ppo.py`
- **Description**: On-policy algorithm with clipped objective for stable training
- **Architecture**: Actor-Critic with CNN feature extractor (Conv2D > Linear layers)
- **Features**: 
  - Clipped surrogate objective for stable updates
  - Multiple epochs per update (4 epochs)
  - Entropy regularization for exploration
  - Advantage estimation for policy improvement
- **Best for**: Both discrete and continuous action spaces, stable learning

## Environments

### Discrete Action Space Environment (`exploration_env.py`)
- **Grid Size**: 100×100 binary grid map
- **View Size**: 5×5 local observation window (partial observability)
- **Actions**: Discrete (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
- **Reward Structure**:
  - **DQN/A2C**: +10 for new cells, -1 for revisits, -100 for collisions
  - **PPO**: +1 for new cells, -10 for collisions
  - Episode ends when all accessible areas explored or max steps reached
- **Features**: 
  - Perlin noise-based map generation for realistic terrain
  - Flood fill algorithm for connectivity validation (>60% free space)
  - Local observation window for partial observability
  - Exploration coverage tracking

### Continuous Action Space Environment (`exploration_env_continuous.py`)
- **Action Space**: Continuous [-1,1]^2 movement vectors (dx,dy)
- **Features**:
  - Smooth movement dynamics with vector-based navigation
  - Realistic physics simulation
  - Scaled and rounded movement for grid translation
  - Enhanced reward shaping for continuous control

## Key Features

### Reward Shaping Strategies
- **Continuous Methods (DDPG/TD3)**:
  - +5 for exploring new free cells
  - -0.1 for revisiting already seen cells
  - -2 for hitting obstacles
  - -1 for remaining in place
  - +10 × coverage ratio bonus at each step
- **Discrete Methods (DQN/A2C)**:
  - +10 for discovering new cells
  - -1 for revisiting old cells
  - -100 for colliding with walls

### Exploration Algorithms
- **Flood Fill**: Implemented in `flood_fill.py` for systematic exploration and connectivity validation
- **Perlin Noise**: Dynamic environment generation using `perlin_noise.py` for realistic terrain patterns
- **Multiple RL Algorithms**: Compare performance across different approaches
- **Trajectory Visualization**: Color-coded path tracking with gradient visualization

### Environment Features
- **Configurable Grid Sizes**: Adapt to different complexity levels (default 100x100)
- **Dynamic Obstacles**: Using Perlin noise for realistic environment changes
- **Reward Shaping**: Sophisticated reward structures for exploration (+1 for new cells, -10 for collisions)
- **Visualization**: Built-in plotting and analysis tools
- **Partial Observability**: 5x5 local observation window
- **Safe Spawning**: Intelligent agent placement in accessible areas

### Training Features
- **Hyperparameter Tuning**: Easy configuration for different algorithms
- **Progress Monitoring**: Real-time training progress and metrics
- **Model Checkpointing**: Save and load trained models
- **Performance Analysis**: Comprehensive evaluation metrics

## Performance Metrics & Results
The project tracks various performance metrics and provides comprehensive analysis:

### Key Results
- **TD3 Performance**: Achieves over 80% coverage in best cases with faster convergence
- **DDPG Performance**: Shows consistent improvements with reward shaping
- **Discrete Methods**: DQN and A2C show limited exploration with high variance
- **Training Stability**: TD3 demonstrates lower variance and smoother learning curves

### Metrics Tracked
- **Exploration Efficiency**: Percentage of environment explored (coverage %)
- **Path Length**: Optimality of exploration paths
- **Training Stability**: Loss curves and convergence patterns
- **Algorithm Comparison**: Side-by-side performance analysis
- **Reward Tracking**: Total rewards per episode
- **Coverage Progress**: Exploration percentage over time

## Authors

**Hariharan Sureshkumar** - MS Robotics, Computer Science, Northeastern University  
**Ajinkya Bhandare** - MS Robotics, Electrical and Computer Engineering, Northeastern University

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Added amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Contact
For questions or contributions, please open an issue on the repository or contact the maintainers.
