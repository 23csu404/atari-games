import numpy as np

class SimpleBreakout:
    def __init__(self):
        self.width = 200
        self.height = 300
        self.paddle_speed = 10
        self.ball_speed = 5
        
        self.reset()

    def reset(self):
        # Ball position
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        
        # Ball direction (dx, dy)
        self.dx = np.random.choice([-1, 1])
        self.dy = -1

        # Paddle position
        self.paddle_x = self.width // 2
        self.paddle_width = 40

        # Score
        self.score = 0
        
        return self._get_state()

    # Convert continuous positions to discrete state
    def _get_state(self):
        bx = self.ball_x // 40        # 5 bins
        by = self.ball_y // 60        # 5 bins
        dx = 0 if self.dx == -1 else 1
        dy = 0 if self.dy == -1 else 1
        px = self.paddle_x // 40      # 5 bins

        return (bx, by, dx, dy, px)

    # -----------------------------
    # Agent actions:
    # 0 = stay, 1 = left, 2 = right
    # -----------------------------
    def step(self, action):

        # AGENT-CONTROLLED PADDLE
        if action == 1:
            self.paddle_x -= self.paddle_speed
        elif action == 2:
            self.paddle_x += self.paddle_speed

        self.paddle_x = np.clip(self.paddle_x, 0, self.width)

        # Move ball
        self.ball_x += self.dx * self.ball_speed
        self.ball_y += self.dy * self.ball_speed
        
        # Wall bounce
        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.dx *= -1

        if self.ball_y <= 0:
            self.dy *= -1

        # Paddle hit
        if self.ball_y >= self.height - 20:
            if abs(self.ball_x - self.paddle_x) < self.paddle_width:
                self.dy *= -1
                reward = +1      # good hit
                done = False
            else:
                reward = -10     # agent missed ball
                done = True
                return self._get_state(), reward, done

        else:
            reward = 0
            done = False

        return self._get_state(), reward, done
