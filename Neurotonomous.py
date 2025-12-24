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



