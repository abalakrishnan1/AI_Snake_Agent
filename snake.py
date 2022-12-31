from collections import deque
import os
import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 
from enum import Enum

# constants
class direction(Enum):
    UP :int = 1
    DOWN :int = 2
    LEFT :int = 3
    RIGHT :int = 4


class SnakeBody:
    dir = direction.UP
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction
        
        
SCREEN_SIZE :tuple = (400, 435)
UNIT_SIZE :int = 20
STARTING_POINT :int = 200

# colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

#ml constants
BATCH_SIZE = 1000

snake_body :list[list] = [SnakeBody(*[STARTING_POINT, STARTING_POINT, direction.UP]),
                          SnakeBody(*[STARTING_POINT, STARTING_POINT + UNIT_SIZE, direction.UP]),
                          SnakeBody(*[STARTING_POINT, STARTING_POINT + 2*UNIT_SIZE, direction.UP])]

# game AI vars
memory = deque(maxlen=100000)
done = False
n_game = 0
reward = 0
max_score = 0
msg = ""

curr_dir :direction = direction.UP

#pygame init
pygame.init()
myfont = pygame.font.SysFont("monospace", 16)
score_rect = pygame.Rect(0, 400, 400, 35)

running :bool = True

# pygame config screen
screen :pygame.Surface = pygame.display.set_mode(SCREEN_SIZE)
pygame.draw.rect(screen, WHITE, score_rect)

# init render
for i in range(len(snake_body)):
    pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
score = myfont.render("Score: {0}".format(0), 1, BLACK)
screen.blit(score, (10, 400))
pygame.display.update()

clock :pygame.time.Clock = pygame.time.Clock()

food_position :np.array = np.array([np.random.randint(0, 19) * 20, np.random.randint(0, 19) * 20])
         
def reset():
    global done, reward, snake_body, food_position, msg, curr_dir
    reward = False
    done = False
    curr_dir = direction.UP
    food_position = np.array([np.random.randint(0, 19) * 20, np.random.randint(0, 19) * 20])
    snake_body = [SnakeBody(*[STARTING_POINT, STARTING_POINT, direction.UP]),
                          SnakeBody(*[STARTING_POINT, STARTING_POINT + UNIT_SIZE, direction.UP]),
                          SnakeBody(*[STARTING_POINT, STARTING_POINT + 2*UNIT_SIZE, direction.UP])]
    
    screen.fill(BLACK)
    for i in range(len(snake_body)):
        pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
    score = myfont.render("Score: {0}".format(0), 1, BLACK)
    screen.blit(score, (10, 400))
    pygame.display.update()


def check_body_will_collide(head_future):
    for i in range(1, len(snake_body)):
        if head_future.x == snake_body[i].x and head_future.y == snake_body[i].y:
            return True
    return False

def check_will_collide(dir, danger_dir):
    head = snake_body[0]
    if danger_dir == direction.UP: # straight
        if dir == direction.UP:
            if head.y - UNIT_SIZE <= 0 or check_body_will_collide(SnakeBody(head.x, head.y - UNIT_SIZE, curr_dir)):
                return True
        if dir == direction.DOWN:
            if head.y + UNIT_SIZE >= 400 or check_body_will_collide(SnakeBody(head.x, head.y + UNIT_SIZE, curr_dir)):
                return True
        if dir == direction.RIGHT:
            if head.x + UNIT_SIZE >= 400 or check_body_will_collide(SnakeBody(head.x + UNIT_SIZE, head.y, curr_dir)):
                return True
        if dir == direction.LEFT:
            if head.x - UNIT_SIZE < 0 or check_body_will_collide(SnakeBody(head.x - UNIT_SIZE, head.y, curr_dir)):
                return True
        return False
    elif danger_dir == direction.LEFT: # left
        if dir == direction.UP:
            if head.x - UNIT_SIZE < 0 or check_body_will_collide(SnakeBody(head.x - UNIT_SIZE, head.y, curr_dir)):
                return True
        if dir == direction.DOWN:
            if head.x + UNIT_SIZE >= 400 or check_body_will_collide(SnakeBody(head.x + UNIT_SIZE, head.y, curr_dir)):
                return True
        if dir == direction.RIGHT:
            if head.y - UNIT_SIZE < 0 or check_body_will_collide(SnakeBody(head.x, head.y - UNIT_SIZE, curr_dir)):
                return True
        if dir == direction.LEFT:
            if head.y + UNIT_SIZE >= 400 or check_body_will_collide(SnakeBody(head.x, head.y + UNIT_SIZE, curr_dir)):
                return True
        return False
    elif danger_dir == direction.RIGHT: #right
        if dir == direction.UP:
            if head.x + UNIT_SIZE >= 400 or check_body_will_collide(SnakeBody(head.x + UNIT_SIZE, head.y, curr_dir)):
                return True
        if dir == direction.DOWN:
            if head.x - UNIT_SIZE < 0 or check_body_will_collide(SnakeBody(head.x - UNIT_SIZE, head.y, curr_dir)):
                return True
        if dir == direction.RIGHT:
            if head.y + UNIT_SIZE >= 400 or check_body_will_collide(SnakeBody(head.x, head.y + UNIT_SIZE, curr_dir)):
                return True
        if dir == direction.LEFT:
            if head.y - UNIT_SIZE < 0 or check_body_will_collide(SnakeBody(head.x, head.y - UNIT_SIZE,curr_dir)):
                return True
        return False

def get_state():
    global curr_dir
    head = snake_body[0]
    
    state = [
        # Danger Straight
        check_will_collide(curr_dir, direction.UP),
 
        # Danger right
        check_will_collide(curr_dir, direction.RIGHT),
 
        # Danger Left
        check_will_collide(curr_dir, direction.LEFT),
 
        # current direction one-hot encoding
        curr_dir == direction.LEFT,
        curr_dir == direction.RIGHT,
        curr_dir == direction.UP,
        curr_dir == direction.DOWN,
        
        # food relative to head
        food_position[0] < head.x,
        food_position[0] > head.x,
        food_position[1] < head.y,
        food_position[1] > head.y,
        
        np.sqrt((head.x - food_position[0])**2 + (head.y - food_position[1])**2)
        
    ]
    
    return np.array(state, dtype=int)

# ml classes and functionality
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)
        self.linear3 = nn.Linear(hidden_size//2, output_size)
 
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x
 
    def save(self, file_name='model_name.pth'):
        model_folder_path = 'Path'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        #Learning Rate for Optimizer
        self.lr = lr
        #Discount Rate
        self.gamma = gamma
        #Linear NN defined above.
        self.model = model
        #optimizer for weight and biases updation
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)
        #Mean Squared error loss function
        self.criterion = nn.MSELoss()
 
     
    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
 
        # if only one parameter to train , then convert to tuple of shape (1, x)
        if(len(state.shape) == 1):
            #(1, x)
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )
 
        # 1. Predicted Q value with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue)
        #pred.clone()
        #preds[argmax(action)] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward() # backward propagation of loss
 
        self.optimer.step()

model = Linear_QNet(12, 256, 3)
trainer = QTrainer(model, lr = 0.001, gamma = 0.9)

def get_action(state):
    # random moves: tradeoff explotation / exploitation
    epsilon = 100 - n_game
    final_move = [0, 0, 0]
    if(random.randint(0, 200) < epsilon):
        move = random.randint(0, 2)
        final_move[move] = 1
    else:
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = model(state0)  # prediction by model
        move = torch.argmax(prediction).item()
        final_move[move] = 1
    return final_move

def remember(state,action,reward,next_state,done):
    memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

def train_long_memory():

    if (len(memory) > BATCH_SIZE):
        mini_sample = random.sample(memory,BATCH_SIZE)
    else:
        mini_sample = memory
    states,actions,rewards,next_states,dones = zip(*mini_sample)
    trainer.train_step(states,actions,rewards,next_states,dones)

def train_short_memory(state,action,reward,next_state,done):
    trainer.train_step(state,action,reward,next_state,done)

def capture_food():
    global food_position, reward
    if snake_body[0].x == food_position[0] and snake_body[0].y == food_position[1]:
        if snake_body[len(snake_body) - 1].direction == direction.RIGHT:
            snake_body.append(SnakeBody(*[snake_body[len(snake_body)-1].x - UNIT_SIZE, snake_body[len(snake_body)-1].y, direction.RIGHT]))
        elif snake_body[len(snake_body) - 1].direction == direction.LEFT:
            snake_body.append(SnakeBody(*[snake_body[len(snake_body)-1].x + UNIT_SIZE, snake_body[len(snake_body)-1].y, direction.LEFT]))
        elif snake_body[len(snake_body) - 1].direction == direction.UP:
            snake_body.append(SnakeBody(*[snake_body[len(snake_body)-1].x, snake_body[len(snake_body)-1].y + UNIT_SIZE, direction.UP]))
        elif snake_body[len(snake_body) - 1].direction == direction.DOWN:
            snake_body.append(SnakeBody(*[snake_body[len(snake_body)-1].x, snake_body[len(snake_body)-1].y - UNIT_SIZE, direction.DOWN]))
        food_position = np.array([np.random.randint(0, 19) * 20, np.random.randint(0, 19) * 20])
        reward += 30

# if snake is out of bounds
def check_window_collision():
    global done, reward, msg
    if snake_body[0].x < 0 or snake_body[0].x >= 400 or snake_body[0].y < 0 or snake_body[0].y >= 400:
        done = True
        msg = "window!"
        reward -= 10

def check_body_collision():
    global done, reward, msg
    for i in range(1, len(snake_body)):
        if snake_body[i].x == snake_body[0].x and snake_body[i].y == snake_body[0].y:
            done = True
            msg = "body!"
            reward -= 10
            
def move_snake(move):
    global max_score, msg

    if move[0] == 1: curr_dir = direction.UP
    elif move[1] == 1: curr_dir = direction.RIGHT
    elif move[2] == 1: curr_dir = direction.LEFT
        
    for i in range(len(snake_body)):
        if i == 0: snake_body[i].direction = curr_dir
        if snake_body[i].direction == direction.UP:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
            snake_body[i].y -= UNIT_SIZE 
            pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
            
        if snake_body[i].direction == direction.DOWN:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
            snake_body[i].y += UNIT_SIZE 
            pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
            
        if snake_body[i].direction == direction.RIGHT:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
            snake_body[i].x += UNIT_SIZE 
            pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
            
        if snake_body[i].direction == direction.LEFT:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))
            snake_body[i].x -= UNIT_SIZE 
            pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE))    
    
    for i in range(len(snake_body)):
        if i != 0: snake_body[len(snake_body) - i].direction = snake_body[len(snake_body) - i - 1].direction

    pygame.draw.rect(screen, WHITE, score_rect)
    max_score = max(max_score, len(snake_body)-3)
    score = myfont.render("Max Score: {0}".format(max_score), 1, BLACK)
    count = myfont.render("Game #: {0}".format(n_game), 1, BLACK)
    msg_screen = myfont.render(msg, 1, BLACK)
    screen.blit(score, (10, 400))
    screen.blit(count, (10, 415))
    screen.blit(msg_screen, (200, 400))
    pygame.draw.rect(screen, RED, pygame.Rect(food_position[0], food_position[1], UNIT_SIZE, UNIT_SIZE))
    
    pygame.display.update()
        
# game loop
while running:
    reward = 0
    
    state_old = get_state()
    move = get_action(state_old)
    
    move_snake(move)
        
    state_new = get_state()
    
    capture_food()
    check_window_collision()
    check_body_collision()
    
    train_short_memory(state_old, move, reward, state_new, done)
    remember(state_old, move, reward, state_new, done)
    
    print(state_new)
    if done:
        n_game += 1
        train_long_memory()
        reset()
    
    clock.tick(100)