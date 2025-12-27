# Neurotonomous

**Neurotonomous** is a 2D autonomous driving simulator built in Python using **PyTorch** and **Pygame**. It trains a neural network to drive a car and avoid randomly generated obstacles using only 5 ray-cast sensors and **Deep Q-Learning (DQN)** (there are no hardcoded rules, no human driving data).

The agent starts with completely random actions and, through trial-and-error reinforcement learning, gradually learns to navigate complex obstacle fields.
*(Green circle = car | Red lines = sensors | White rectangles = obstacles)*

## Features
- **Deep Q-Network (DQN)** implemented with PyTorch (5→24→24→3 architecture)
- **Experience replay** buffer (10,000 capacity) for stable training
- **Epsilon-greedy exploration** with decay for balanced learning
- Real-time visualization with Pygame
- 5 forward-facing ray-cast sensors (150px range)
- Procedural random obstacle generation every episode
- Toroidal (wrap-around) world
- Live on-screen metrics: episode, steps survived, epsilon
- Post-training demo mode showcasing learned behavior

## Demo
After ~500 training episodes (or ~30 in quick mode), the car learns to:
- Avoid walls proactively
- Turn toward open spaces
- Survive hundreds of steps in dense random layouts

Early episodes: crashes in <50 steps  
Trained agent: often survives 500–1000+ steps

## Requirements

- Python 3.8+
- PyTorch
- Pygame
- NumPy
