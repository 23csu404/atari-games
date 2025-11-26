🎮 Breakout Game – Reinforcement Learning Agent

A Q-Learning Based Autonomous Game Player

📌 Overview

This project implements a Reinforcement Learning (RL) agent that learns to play the classic Breakout arcade game autonomously.
The environment is built using Python + Pygame, while the agent is trained using the Q-Learning algorithm.
The agent observes the environment, takes actions, receives rewards, and gradually learns to hit bricks and avoid losing the ball.

This repository includes:

Custom Pygame Breakout environment

Q-Learning agent

State discretization

Reward function design

Training loop & performance graphs

Autonomous gameplay mode

Q-table saving & reloading

✨ Features

🧠 Custom RL Agent (Q-Learning)

🎮 Smooth Breakout Environment using Pygame

⭐ Improved paddle movement with smoothing

📊 Training visualizations (Rewards, Bricks, Lives)

📁 Modular code structure (Environment, Agent, Training, Gameplay)

💾 Q-table persistence (q_table.pkl)

🚀 Autonomous gameplay demonstration

📂 Project Structure
├── environment.py        # Breakout game environment (Pygame)
├── agent.py              # Q-learning agent implementation
├── train.py              # Training script
├── play.py               # Run autonomous gameplay with trained agent
├── graphs/               # Training visualizations
├── q_table.pkl           # Saved Q-table after training
└── README.md             # Project documentation

🕹️ Game Environment Design

The custom environment includes:

Paddle

Moves left/right

Smooth movement reduces jitter

Controlled entirely by agent

Ball

Continuous physics

Collisions with paddle, bricks, walls

Bricks

Multiple rows

Destroyed upon collision

Provide reward

State Representation (Discrete):
(ball_x_bin, ball_y_bin, ball_dx_bin, ball_dy_bin, paddle_x_bin)

Action Space
Action	Meaning
0	Stay
1	Move Left
2	Move Right
Reward Structure

+10 → Hit a brick

+2 → Ball hits paddle

−20 → Lose ball (life lost)

+50 → Clear all bricks

Optional small negative reward for delaying the game

🤖 Q-Learning Algorithm

The agent updates its Q-values using the Bellman equation:

Q(s, a) = Q(s, a) + α [ r + γ max(Q(s', :)) – Q(s, a) ]


✔ Off-policy
✔ Fast for discrete states
✔ Works well for medium-sized RL tasks like Breakout

📈 Training Process

Training involves:

Reset environment

Choose action (epsilon-greedy)

Observe reward + next state

Update Q-table

Decay exploration

Save q_table.pkl

Performance graphs include:

Average reward per episode

Bricks destroyed

Lives remaining

Paddle-ball interaction quality

🎥 Autonomous Gameplay

Once trained, run:

python play.py


The agent will:

Track the ball accurately

Move the paddle smoothly

Destroy bricks efficiently

Play the full game with no human input

🧪 Challenges & Solutions
❌ Paddle jitter

✔ Added smoothing + better discretization

❌ Q-table not converging

✔ Tuned learning rate, gamma, reward structure

❌ Poor collision detection

✔ Improved bounce logic + angle handling

❌ Agent stuck in loops

✔ Modified exploration & reward shaping
