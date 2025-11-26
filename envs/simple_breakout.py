import pygame
import sys
import random
import numpy as np
import pickle
from pathlib import Path

# --------------------------------------------------------
# Smooth Breakout Environment (No vibration, stable agent)
# --------------------------------------------------------

class SmoothBreakoutEnv:
    def __init__(self, width=600, height=700):
        self.width = width
        self.height = height

        # Paddle
        self.paddle_w = 100
        self.paddle_h = 12
        self.paddle_x = width // 2
        self.paddle_speed = 6
        self.paddle_dx = 0  # paddle velocity for smoothing

        # Ball
        self.ball_x = width // 2
        self.ball_y = height // 2
        self.ball_r = 7
        self.ball_speed = 5
        self.dx = random.choice([-1, 1])
        self.dy = -1

        # Bricks
        self.rows = 5
        self.cols = 8
        self.bricks = []
        self.brick_w = (self.width - (self.cols + 1) * 6) // self.cols
        self.brick_h = 25
        self.build_bricks()

        # Discretization bins (improved resolution)
        self.bx_bins = 8
        self.by_bins = 8
        self.px_bins = 8

    def build_bricks(self):
        self.bricks.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                x = 6 + c * (self.brick_w + 6)
                y = 60 + r * (self.brick_h + 6)
                self.bricks.append(pygame.Rect(x, y, self.brick_w, self.brick_h))

    def reset(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.dx = random.choice([-1, 1])
        self.dy = -1
        self.paddle_x = self.width // 2
        self.paddle_dx = 0
        return self.get_state()

    def get_state(self):
        bx = min(self.bx_bins - 1, int(self.ball_x / (self.width / self.bx_bins)))
        by = min(self.by_bins - 1, int(self.ball_y / (self.height / self.by_bins)))
        px = min(self.px_bins - 1, int(self.paddle_x / (self.width / self.px_bins)))
        dx = 0 if self.dx < 0 else 1
        dy = 0 if self.dy < 0 else 1
        return (bx, by, dx, dy, px)

    def step(self, action):
        # ACTIONS = 0 stay, 1 left, 2 right
        if action == 1:
            target = self.paddle_x - 40
        elif action == 2:
            target = self.paddle_x + 40
        else:
            target = self.paddle_x

        # ✔ Smooth paddle movement
        self.paddle_x += (target - self.paddle_x) * 0.25

        # Clamp paddle
        self.paddle_x = max(self.paddle_w//2, min(self.width - self.paddle_w//2, self.paddle_x))

        # Move ball
        self.ball_x += self.dx * self.ball_speed
        self.ball_y += self.dy * self.ball_speed

        reward = 0
        done = False

        # Wall
        if self.ball_x <= self.ball_r:
            self.ball_x = self.ball_r
            self.dx *= -1
        if self.ball_x >= self.width - self.ball_r:
            self.ball_x = self.width - self.ball_r
            self.dx *= -1
        if self.ball_y <= self.ball_r:
            self.ball_y = self.ball_r
            self.dy *= -1

        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x - self.paddle_w//2,
                                  self.height - 40,
                                  self.paddle_w, self.paddle_h)
        ball_rect = pygame.Rect(self.ball_x - self.ball_r,
                                self.ball_y - self.ball_r,
                                2 * self.ball_r, 2 * self.ball_r)

        if ball_rect.colliderect(paddle_rect) and self.dy > 0:
            self.dy *= -1
            reward += 2

        # Brick collision
        for b in self.bricks[:]:
            if ball_rect.colliderect(b):
                self.bricks.remove(b)
                reward += 10
                self.dy *= -1
                break

        # Ball lost
        if self.ball_y > self.height:
            reward -= 20
            done = True

        # Win condition
        if not self.bricks:
            reward += 50
            done = True

        return self.get_state(), reward, done

    def render(self, screen):
        screen.fill((0, 0, 0))

        # Bricks
        for b in self.bricks:
            pygame.draw.rect(screen, (0, 200, 0), b)

        # Paddle
        pygame.draw.rect(screen, (50, 150, 255),
                         pygame.Rect(self.paddle_x - self.paddle_w//2,
                                     self.height - 40,
                                     self.paddle_w, self.paddle_h))

        # Ball
        pygame.draw.circle(screen, (255, 80, 80),
                           (int(self.ball_x), int(self.ball_y)), self.ball_r)

# --------------------------------------------------------
# Run Agent (smooth)
# --------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 700))
    pygame.display.set_caption("Smooth Agent Breakout")
    clock = pygame.time.Clock()

    env = SmoothBreakoutEnv()
    state = env.reset()

    # Load Q-table
    try:
        Q = pickle.load(open("q_table.pkl", "rb"))
        epsilon = 0.0
        print("Loaded trained Q-table.")
    except:
        print("No Q-table found. Running random agent.")
        Q = None
        epsilon = 1.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agent decides action
        if Q is not None and random.random() > epsilon:
            action = np.argmax(Q[state])
        else:
            action = random.choice([0, 1, 2])

        next_state, reward, done = env.step(action)

        env.render(screen)
        pygame.display.flip()
        clock.tick(60)

        state = next_state
        if done:
            pygame.time.wait(1200)
            state = env.reset()

    pygame.quit()

if __name__ == "__main__":
    main()

