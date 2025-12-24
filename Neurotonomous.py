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

