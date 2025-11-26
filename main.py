import numpy as np
from envs.simple_breakout import SimpleBreakout
from agents.q_learning_agent import QLearningAgent
from utils.plot_utils import plot_rewards

# Initialize environment
env = SimpleBreakout()
state_shape = (5, 5, 2, 2, 5)   # (ball_x, ball_y, dx, dy, paddle_x)
n_actions = 3                   # 0=stay, 1=left, 2=right

agent = QLearningAgent(state_shape, n_actions)
episodes = 1000
rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for _ in range(1000):  # maximum steps per episode
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        total_reward += reward
        state = next_state
        if done:
            break

    agent.decay_epsilon()
    rewards.append(total_reward)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward} | Epsilon: {agent.epsilon:.2f}")

plot_rewards(rewards)
