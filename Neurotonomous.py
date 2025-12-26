import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque

WIDTH, HEIGHT = 800, 600
CAR_RADIUS = 10 # car is drawn as a circle
SENSOR_ANGLES = [-60, -30, 0, 30, 60]  # 5 sensors with degrees relative to car angle
SENSOR_LENGTH = 150
NUM_SENSORS = len(SENSOR_ANGLES)
ACTIONS = ['left', 'right', 'nothing']  # 0: left, 1: right, 2: nothing
TURN_RATE = 3  # constants for turning and acceleration
ACCEL = 0.2
MAX_SPEED = 5
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FONT_COLOR = (200, 255, 200)

class QNetwork(nn.Module): # Q-Network for 5 sensors and 3 actions (PyTorch)
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(NUM_SENSORS, 24) # 5 sensors with 24 hidden neurons - first layer 
        self.fc2 = nn.Linear(24, 24) # feeds first layer of 24 neurons to second layer of 24 hidden neurons - second layer
        self.fc3 = nn.Linear(24, len(ACTIONS)) # feeds second layer of 24 neurons to output layer with 3 neurons for each action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) # relu takes max(0, x) for each element in tensor and sets negative values to 0
        return self.fc3(x) # output layer with highest Q-value is the best Q-action to take

class Car: # this class represents the AI car and its actions
    def __init__(self, x, y, angle=0):
        self.x = x # constructor for new car
        self.y = y
        self.angle = angle  
        self.speed = 0

    def accelerate(self): # called to increase speed (no braking system)
        self.speed = min(self.speed + ACCEL, MAX_SPEED)

    def turn_left(self): # called for turning left
        self.angle -= TURN_RATE

    def turn_right(self): # called for turning right
        self.angle += TURN_RATE

    def update(self): # called to update car position based on angle and speed
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        self.x %= WIDTH # wrap around screen edges
        self.y %= HEIGHT

    def get_sensor_readings(self, screen, obstacles):
        readings = []
        for offset in SENSOR_ANGLES: 
            dist = SENSOR_LENGTH
            for d in range(1, SENSOR_LENGTH + 1, 5): # casts a ray forward - checking pixels every 5 steps up to 150 pixels
                rad = math.radians(self.angle + offset)
                sx = int(self.x + d * math.cos(rad))
                sy = int(self.y + d * math.sin(rad))
                if 0 <= sx < WIDTH and 0 <= sy < HEIGHT: 
                    color = screen.get_at((sx, sy))[:3]
                    if color == WHITE: # as soon as it hits a white pixel/obstacle it stops and records how far it got
                        dist = d
                        break
            readings.append(dist / SENSOR_LENGTH) # returns a list of 5 numbers between 0 and 1 (1 = obstacle, 0 = nothing) 
                                                # the 5 numbers are what the QNetwork uses as input
        return readings

def generate_obstacles(num=10): # randomly generates 10 white rectangles (obstacles) on the screen
    obstacles = []
    for _ in range(num):
        w, h = random.randint(20, 50), random.randint(20, 50)
        x, y = random.randint(0, WIDTH - w), random.randint(0, HEIGHT - h)
        obstacles.append(pygame.Rect(x, y, w, h))
    return obstacles

def is_crashed(car, obstacles):
    car_rect = pygame.Rect(car.x - CAR_RADIUS, car.y - CAR_RADIUS, CAR_RADIUS * 2, CAR_RADIUS * 2) # rectangle created around the car to act as bounding box
    return car_rect.collidelist(obstacles) != -1 # checks if the car has collided with any of the obstacles and square overlaps with any of the obstacles

def train(model, optimizer, replay_buffer, batch_size=32, gamma=0.95): # neural network brain function to train the Q-Network
    if len(replay_buffer) < batch_size: # batch size of 32 learning experiences, if not enough experiences yet then dont train
        return
    batch = random.sample(replay_buffer, batch_size) # randomly samples 32 experiences from the replay buffer
    states, actions, rewards, next_states = zip(*batch) # unzips the batch into 4 separate lists
    states = torch.tensor(np.array(states), dtype=torch.float32) # converts lists to PyTorch tensors
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    q_values = model(states) # predicts Q-values for the 3 actions given the current states
    next_q_values = model(next_states)
    max_next_q = torch.max(next_q_values, dim=1)[0] 
    target_q = q_values.clone()
    updates = rewards + gamma * max_next_q # correct Q-value = immediate reward + discounted best future reward
    target_q[range(batch_size), actions] = updates # only update the Q-value for the action that was actually taken

    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad() # clear old gradients.
    loss.backward() # compute how to adjust weights to reduce error
    optimizer.step() # update weights

def draw_text(screen, text, size, x, y, color=FONT_COLOR): # helper for displaying text on the screen (shows information to user in a clean way)
    font = pygame.font.SysFont('Arial', size, bold=True)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)


def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT))  
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)  
    pygame.display.set_caption("Neurotonomous")
    clock = pygame.time.Clock()

    title = True
    while title: # screen loop for title, press any key to start training, and quit button to exit
        screen.fill(BLACK)
        draw_text(screen, "Neurotonomous", 72, WIDTH // 2, HEIGHT // 3)
        draw_text(screen, "AI Driving Simulator", 36, WIDTH // 2, HEIGHT // 2)
        draw_text(screen, "Press any key to start training", 28, WIDTH // 2, HEIGHT * 2 // 3)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                title = False

    model = QNetwork() # create Q-Network brain
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer
    replay_buffer = deque(maxlen=10000) # replay buffer to store 10000 learning experiences
    epsilon = 1.0 # starts with full random actions then becomes less random over time
    epsilon_decay = 0.995
    max_episodes = 500

    print("Neurotonomous - Training started...")

    episode = 0
    while episode < max_episodes:
        car = Car(WIDTH // 2, HEIGHT // 2, random.randint(0, 360))
        obstacles = generate_obstacles()
        steps = 0
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get(): # handles quit option
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill(BLACK) 
            for obs in obstacles: # draws obstacles
                pygame.draw.rect(screen, WHITE, obs)

            state = car.get_sensor_readings(screen, obstacles)

            if random.random() < epsilon: # with early training, car takes random actions to explore environment but over time it relies more on using what the model has learned
                action = random.randint(0, len(ACTIONS) - 1) # 0 = left, 1 = right, 2 = nothing
            else:
                with torch.no_grad(): # uses the Q-Network to predict the best action based on current sensor readings, converts state to tensor first
                    q_vals = model(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_vals).item()

            if action == 0: # performs the selected action
                car.turn_left()
            elif action == 1:
                car.turn_right()
            car.accelerate()
            car.update()

            pygame.draw.circle(screen, GREEN, (int(car.x), int(car.y)), CAR_RADIUS) # draws the car as a green circle, and sensor lines
            for i, offset in enumerate(SENSOR_ANGLES):
                rad = math.radians(car.angle + offset)
                sx = int(car.x + state[i] * SENSOR_LENGTH * math.cos(rad))
                sy = int(car.y + state[i] * SENSOR_LENGTH * math.sin(rad))
                pygame.draw.line(screen, RED, (car.x, car.y), (sx, sy), 2)

            draw_text(screen, f"Episode: {episode}  Steps: {steps}  Epsilon: {epsilon:.3f}", 24, WIDTH // 2, 20)

            pygame.display.flip()

            next_state = car.get_sensor_readings(screen, obstacles) # gets next state after action
            crashed = is_crashed(car, obstacles) 
            reward = -100 if crashed else 1 - np.mean(state) # negative reward for crashing, positive reward for distance from obstacles
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state)) # stores experience in replay buffer
            train(model, optimizer, replay_buffer) # trains the Q-Network with a batch of experiences from the replay buffer

            done = crashed or steps > 1000 # ends episode if crashed or max steps reached
            steps += 1
            clock.tick(60)

        epsilon = max(0.1, epsilon * epsilon_decay) # decays epsilon after each episode
        print(f"Episode {episode}: Reward {total_reward:.1f}, Steps {steps}, Epsilon {epsilon:.3f}")
        episode += 1

    print("Training finished, running Neurotonomous demo...")

    car = Car(WIDTH // 2, HEIGHT // 2) # reset car for demo - no randomness, always picks best action
    obstacles = generate_obstacles()
    running = True
    while running:
        for event in pygame.event.get(): # handles quit option
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        for obs in obstacles:
            pygame.draw.rect(screen, WHITE, obs)

        state = car.get_sensor_readings(screen, obstacles) # gets current sensor readings
        with torch.no_grad():
            q_vals = model(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_vals).item()

        if action == 0: # performs the selected action
            car.turn_left()
        elif action == 1:
            car.turn_right()
        car.accelerate()
        car.update()

        pygame.draw.circle(screen, GREEN, (int(car.x), int(car.y)), CAR_RADIUS) # draws the car and sensor lines
        for i, offset in enumerate(SENSOR_ANGLES):
            rad = math.radians(car.angle + offset)
            sx = int(car.x + state[i] * SENSOR_LENGTH * math.cos(rad))
            sy = int(car.y + state[i] * SENSOR_LENGTH * math.sin(rad))
            pygame.draw.line(screen, RED, (car.x, car.y), (sx, sy), 2)

        draw_text(screen, "Neurotonomous - Demo (Trained Model)", 32, WIDTH // 2, 30)
        draw_text(screen, "Close window to exit", 24, WIDTH // 2, HEIGHT - 30)

        pygame.display.flip()

        if is_crashed(car, obstacles): # checks for crash
            draw_text(screen, "CRASHED!", 80, WIDTH // 2, HEIGHT // 2, (255, 50, 50)) 
            pygame.display.flip()
            pygame.time.wait(2000)
            running = False

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

