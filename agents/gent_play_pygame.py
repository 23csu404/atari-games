"""
Agent-playable Breakout (Pygame)
- Agent controls paddle automatically (no keyboard control)
- Loads Q-table from 'q_table.pkl' if available
- If not available, agent acts randomly (epsilon=1)
Requires: pygame, numpy
"""

import pygame
import sys
import random
import numpy as np
import pickle
from pathlib import Path

# Try import your QLearningAgent implementation
try:
    from agents.q_learning_agent import QLearningAgent
except Exception:
    # Minimal fallback QLearningAgent if your module isn't found.
    class QLearningAgent:
        def __init__(self, state_shape=(5,5,2,2,5), n_actions=3):
            self.Q = np.zeros(state_shape + (n_actions,))
            self.actions = list(range(n_actions))
            self.epsilon = 1.0
        def choose_action(self, state):
            # state is a tuple of indices
            if random.random() < getattr(self, "epsilon", 1.0):
                return random.choice(self.actions)
            return int(np.argmax(self.Q[state]))
        def decay_epsilon(self): pass

# ------------------------------
# Simple agent-friendly environment with pygame rendering
# ------------------------------
class PygameSimpleBreakout:
    def __init__(self, width=400, height=500, paddle_width=80):
        self.width = width
        self.height = height
        self.paddle_w = paddle_width
        self.paddle_h = 12
        self.ball_r = 7
        self.paddle_speed = 10
        self.ball_speed = 5

        # Discretization bins (must match agent state_shape)
        self._nx = 5   # ball x bins
        self._ny = 5   # ball y bins
        self._p_bins = 5  # paddle x bins

        self.reset()

    def reset(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.dx = random.choice([-1, 1])
        self.dy = -1
        self.paddle_x = self.width // 2
        self.score = 0
        self.lives = 3

        # bricks grid
        self.brick_rows = 4
        self.brick_cols = 7
        self.brick_w = (self.width - (self.brick_cols + 1) * 6) // self.brick_cols
        self.brick_h = 18
        self.bricks = []
        for r in range(self.brick_rows):
            for c in range(self.brick_cols):
                x = 6 + c * (self.brick_w + 6)
                y = 60 + r * (self.brick_h + 6)
                self.bricks.append(pygame.Rect(x, y, self.brick_w, self.brick_h))

        return self._get_state()

    def _get_state(self):
        # discretize ball and paddle positions into small bins
        bx = min(self._nx - 1, int(self.ball_x / (self.width / self._nx)))
        by = min(self._ny - 1, int(self.ball_y / (self.height / self._ny)))
        dx = 0 if self.dx < 0 else 1
        dy = 0 if self.dy < 0 else 1
        px = min(self._p_bins - 1, int(self.paddle_x / (self.width / self._p_bins)))
        return (bx, by, dx, dy, px)

    def step(self, action):
        # action: 0=stay,1=left,2=right (agent-controlled)
        if action == 1:
            self.paddle_x -= self.paddle_speed
        elif action == 2:
            self.paddle_x += self.paddle_speed
        # clamp paddle
        self.paddle_x = max(self.paddle_w//2, min(self.width - self.paddle_w//2, self.paddle_x))

        # move ball
        self.ball_x += self.dx * self.ball_speed
        self.ball_y += self.dy * self.ball_speed

        # wall bounce
        if self.ball_x <= self.ball_r:
            self.ball_x = self.ball_r
            self.dx *= -1
        if self.ball_x >= self.width - self.ball_r:
            self.ball_x = self.width - self.ball_r
            self.dx *= -1
        if self.ball_y <= self.ball_r:
            self.ball_y = self.ball_r
            self.dy *= -1

        reward = 0
        done = False

        # paddle rect for collision
        paddle_rect = pygame.Rect(self.paddle_x - self.paddle_w//2,
                                  self.height - 40,
                                  self.paddle_w, self.paddle_h)

        # check paddle collision
        ball_rect = pygame.Rect(self.ball_x - self.ball_r,
                                 self.ball_y - self.ball_r,
                                 self.ball_r*2, self.ball_r*2)
        if ball_rect.colliderect(paddle_rect) and self.dy > 0:
            self.dy *= -1
            # small reward for safe hit
            reward += 1

            # add angle depending on hit position
            offset = (self.ball_x - self.paddle_x) / (self.paddle_w / 2)
            self.dx = 1 if offset > 0 else -1
            # don't let dx be zero
            if abs(offset) > 0.2:
                # boost horizontal speed influence (no float velocities here)
                pass

        # brick collision
        for b in self.bricks[:]:
            if ball_rect.colliderect(b):
                self.bricks.remove(b)
                reward += 10
                self.dy *= -1
                break

        # ball fell below bottom
        if self.ball_y >= self.height - self.ball_r:
            self.lives -= 1
            reward -= 10
            if self.lives <= 0:
                done = True
            else:
                # reset ball and paddle in center and continue
                self.ball_x = self.width // 2
                self.ball_y = self.height // 2
                self.dx = random.choice([-1, 1])
                self.dy = -1

        # win condition
        if not self.bricks:
            done = True

        return self._get_state(), reward, done

    # rendering helper
    def render(self, surface):
        surface.fill((0,0,0))
        # draw bricks
        for b in self.bricks:
            pygame.draw.rect(surface, (0,180,0), b)
        # draw paddle
        paddle_rect = pygame.Rect(self.paddle_x - self.paddle_w//2,
                                  self.height - 40,
                                  self.paddle_w, self.paddle_h)
        pygame.draw.rect(surface, (50,150,255), paddle_rect)
        # draw ball
        pygame.draw.circle(surface, (255,80,80), (int(self.ball_x), int(self.ball_y)), self.ball_r)
        # draw HUD
        font = pygame.font.SysFont(None, 22)
        surface.blit(font.render(f"Score: { ( (self.brick_rows*self.brick_cols) - len(self.bricks)) * 10 }", True, (255,255,255)), (8,8))
        surface.blit(font.render(f"Lives: {self.lives}", True, (255,255,255)), (self.width-80,8))


# ------------------------------
# Main: create agent and run pygame loop
# ------------------------------
def main():
    pygame.init()
    W, H = 600, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Agent-controlled Breakout (no keyboard control)")
    clock = pygame.time.Clock()

    env = PygameSimpleBreakout(width=W, height=H, paddle_width=100)

    # create agent with same state/action dims used in training
    state_shape = (5,5,2,2,5)
    n_actions = 3
    agent = QLearningAgent(state_shape, n_actions)

    # load Q-table if exists
    q_path = Path("q_table.pkl")
    if q_path.exists():
        try:
            agent.Q = pickle.load(open(q_path, "rb"))
            agent.epsilon = 0.0
            print("Loaded Q-table from q_table.pkl — playing greedily.")
        except Exception as e:
            print("Failed to load q_table.pkl:", e)
            agent.epsilon = 1.0
    else:
        print("No q_table.pkl found. Agent will act randomly (epsilon=1).")
        agent.epsilon = 1.0

    # If your agent implementation uses different attribute names, ensure agent.choose_action(state) works.
    # Convert state to tuple indexes and pass to choose_action.

    state = env.reset()

    # history for optional analysis
    score_history = []
    lives_history = []
    bricks_history = []
    frames = 0

    running = True
    while running:
        frames += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agent chooses action based on discrete state
        action = agent.choose_action(state)  # must accept state tuple indexing into Q
        next_state, reward, done = env.step(action)

        # record metrics
        score_history.append(((env.brick_rows*env.brick_cols) - len(env.bricks)) * 10)
        lives_history.append(env.lives)
        bricks_history.append(len(env.bricks))

        # render and advance
        env.render(screen)
        pygame.display.flip()
        clock.tick(60)

        state = next_state
        if done:
            # show final frame then break
            pygame.time.delay(600)
            running = False

    # end
    print("Episode finished. Final score:", ((env.brick_rows*env.brick_cols) - len(env.bricks)) * 10)
    pygame.quit()

    # Optional: show small plots (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(score_history, label="Score (cumulative)")
        plt.title("Agent Play: Score over Frames")
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10,3))
        plt.plot(lives_history, drawstyle='steps-post', label="Lives")
        plt.title("Agent Play: Lives")
        plt.ylim(-0.5, 3.5)
        plt.xlabel("Frame")
        plt.grid(True)
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
