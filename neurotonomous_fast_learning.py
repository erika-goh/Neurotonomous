import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from enhanced_graphics import EnhancedGraphics

WIDTH, HEIGHT = 800, 600
CAR_RADIUS = 10
SENSOR_ANGLES = [-60, -30, 0, 30, 60]
SENSOR_LENGTH = 150
NUM_SENSORS = len(SENSOR_ANGLES)
ACTIONS = ['left', 'right', 'nothing']
TURN_RATE = 3
ACCEL = 0.2
MAX_SPEED = 5

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Larger network for better learning
        self.fc1 = nn.Linear(NUM_SENSORS, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(ACTIONS))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle  
        self.speed = 0

    def accelerate(self):
        self.speed = min(self.speed + ACCEL, MAX_SPEED)

    def turn_left(self):
        self.angle -= TURN_RATE

    def turn_right(self):
        self.angle += TURN_RATE

    def update(self):
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        self.x %= WIDTH
        self.y %= HEIGHT

    def get_sensor_readings(self, screen, obstacles):
        readings = []
        for offset in SENSOR_ANGLES: 
            dist = SENSOR_LENGTH
            for d in range(1, SENSOR_LENGTH + 1, 5):
                rad = math.radians(self.angle + offset)
                sx = int(self.x + d * math.cos(rad))
                sy = int(self.y + d * math.sin(rad))
                if 0 <= sx < WIDTH and 0 <= sy < HEIGHT: 
                    color = screen.get_at((sx, sy))[:3]
                    if color[0] > 200 and color[1] > 200 and color[2] > 200:
                        dist = d
                        break
            readings.append(dist / SENSOR_LENGTH)
        return readings

def generate_obstacles(num=8):  # Reduced obstacles for easier learning
    obstacles = []
    for _ in range(num):
        w, h = random.randint(30, 60), random.randint(30, 60)
        x, y = random.randint(50, WIDTH - 100), random.randint(50, HEIGHT - 100)
        obstacles.append(pygame.Rect(x, y, w, h))
    return obstacles

def is_crashed(car, obstacles):
    car_rect = pygame.Rect(car.x - CAR_RADIUS, car.y - CAR_RADIUS, CAR_RADIUS * 2, CAR_RADIUS * 2)
    return car_rect.collidelist(obstacles) != -1

def train(model, optimizer, replay_buffer, batch_size=64, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states = zip(*batch)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    q_values = model(states)
    next_q_values = model(next_states)
    max_next_q = torch.max(next_q_values, dim=1)[0]
    target_q = q_values.clone()
    updates = rewards + gamma * max_next_q
    target_q[range(batch_size), actions] = updates

    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)  
    pygame.display.set_caption("Neurotonomous")
    clock = pygame.time.Clock()
    
    graphics = EnhancedGraphics(WIDTH, HEIGHT)

    # Title screen
    title = True
    while title:
        graphics.draw_title_screen(screen)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                title = False

    model = QNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate
    replay_buffer = deque(maxlen=50000)
    epsilon = 1.0
    epsilon_decay = 0.7  # Aggressive decay for fast learning
    epsilon_min = 0.05
    max_episodes = 30
    
    episode = 0
    while episode < max_episodes:
        car = Car(WIDTH // 2, HEIGHT // 2, random.randint(0, 360))
        obstacles = generate_obstacles()
        steps = 0
        total_reward = 0
        done = False
        
        graphics.reset_trail()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill(graphics.colors['background'])
            graphics.draw_background_grid(screen)
            graphics.draw_enhanced_obstacles(screen, obstacles)

            state = car.get_sensor_readings(screen, obstacles)

            # Epsilon-greedy with faster decay
            if random.random() < epsilon:
                action = random.randint(0, len(ACTIONS) - 1)
            else:
                with torch.no_grad():
                    q_vals = model(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_vals).item()

            if action == 0:
                car.turn_left()
            elif action == 1:
                car.turn_right()
            car.accelerate()
            car.update()

            graphics.draw_enhanced_car(screen, car)
            graphics.draw_enhanced_sensors(screen, car, state, SENSOR_ANGLES, SENSOR_LENGTH)
            graphics.draw_stats_panel(screen, episode, steps, epsilon, total_reward)

            pygame.display.flip()

            next_state = car.get_sensor_readings(screen, obstacles)
            crashed = is_crashed(car, obstacles)
            
            # Improved reward system
            if crashed:
                reward = -200  # Heavy penalty for crashing
            else:
                # Reward for staying alive + distance from obstacles
                min_sensor = min(state)
                reward = 2.0 + (min_sensor * 3)  # Bigger reward for being far from obstacles
                
                # Bonus for many steps
                if steps > 50:
                    reward += 1
                if steps > 100:
                    reward += 2
            
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state))
            
            # Train more aggressively early on
            if episode < 10:
                for _ in range(4):  # 4x training in early episodes
                    train(model, optimizer, replay_buffer, batch_size=128, gamma=0.99)
            else:
                for _ in range(2):
                    train(model, optimizer, replay_buffer, batch_size=64, gamma=0.99)

            done = crashed or steps > 1000
            steps += 1
            clock.tick(60)

        # Aggressive epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode}: Reward {total_reward:.1f}, Steps {steps}, Epsilon {epsilon:.3f}")
        episode += 1

    print("Training finished, running Neurotonomous demo...")

    car = Car(WIDTH // 2, HEIGHT // 2)
    obstacles = generate_obstacles()
    graphics.reset_trail()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(graphics.colors['background'])
        graphics.draw_background_grid(screen)
        graphics.draw_enhanced_obstacles(screen, obstacles)

        state = car.get_sensor_readings(screen, obstacles)
        with torch.no_grad():
            q_vals = model(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_vals).item()

        if action == 0:
            car.turn_left()
        elif action == 1:
            car.turn_right()
        car.accelerate()
        car.update()

        graphics.draw_enhanced_car(screen, car)
        graphics.draw_enhanced_sensors(screen, car, state, SENSOR_ANGLES, SENSOR_LENGTH)
        
        graphics.draw_enhanced_text(screen, "Neurotonomous - Demo (Trained Model)", 32, 
                                   WIDTH // 2, 30, graphics.colors['car_accent'])
        graphics.draw_enhanced_text(screen, "Close window to exit", 24, 
                                   WIDTH // 2, HEIGHT - 30, graphics.colors['text_secondary'])

        pygame.display.flip()

        if is_crashed(car, obstacles):
            graphics.draw_crash_effect(screen, car)
            pygame.display.flip()
            pygame.time.wait(2000)
            running = False

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()