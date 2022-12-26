import pygame
import sys
import numpy as np
import tensorflow as tf
# import tf_agents.networks.q_network as q_network
# from tf_agents.agents.dqn import dqn_agent

from enum import Enum

# constants
class direction(Enum):
    UP :int = 1
    DOWN :int = 2
    LEFT :int = 3
    RIGHT :int = 4   

SCREEN_SIZE :tuple = (400, 400)
UNIT_SIZE :int = 20
STARTING_POINT :int = 200

# colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)

snake_body :list[list] = [[STARTING_POINT, STARTING_POINT, direction.UP],
                          [STARTING_POINT, STARTING_POINT + UNIT_SIZE, direction.UP],
                          [STARTING_POINT, STARTING_POINT + 2*UNIT_SIZE, direction.UP]]

curr_dir :direction = direction.UP

pygame.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)

running :bool = True

screen :pygame.Surface = pygame.display.set_mode(SCREEN_SIZE)

# init render
for i in range(len(snake_body)):
    pygame.draw.rect(screen, WHITE, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
pygame.display.update()

clock :pygame.time.Clock = pygame.time.Clock()

food_position :np.array = np.array([np.random.randint(0, 19) * 20, np.random.randint(0, 19) * 20])

def capture_food():
    global food_position
    if snake_body[0][0] == food_position[0] and snake_body[0][1] == food_position[1]:
        if snake_body[len(snake_body) - 1][2] == direction.RIGHT:
            snake_body.append([snake_body[len(snake_body)-1][0] - UNIT_SIZE, snake_body[len(snake_body)-1][1], direction.RIGHT])
        elif snake_body[len(snake_body) - 1][2] == direction.LEFT:
            snake_body.append([snake_body[len(snake_body)-1][0] + UNIT_SIZE, snake_body[len(snake_body)-1][1], direction.LEFT])
        elif snake_body[len(snake_body) - 1][2] == direction.UP:
            snake_body.append([snake_body[len(snake_body)-1][0], snake_body[len(snake_body)-1][1] + UNIT_SIZE, direction.UP])
        elif snake_body[len(snake_body) - 1][2] == direction.DOWN:
            snake_body.append([snake_body[len(snake_body)-1][0], snake_body[len(snake_body)-1][1] - UNIT_SIZE, direction.DOWN])
        food_position = np.array([np.random.randint(0, 19) * 20, np.random.randint(0, 19) * 20])

# if snake is out of bounds
def check_window_collision():
    if snake_body[0][0] <= 0 or snake_body[0][0] >= 400 or snake_body[0][0] <= 0 or snake_body[0][1] >= 400:
        global running
        running = False

def check_body_collision():
    for i in range(1, len(snake_body)):
        if snake_body[i][0] == snake_body[0][0] and snake_body[i][1] == snake_body[0][1]:
            global running
            running = False

# game loop
while running:
    capture_food()
    check_window_collision()
    check_body_collision()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and curr_dir != direction.DOWN:
                curr_dir = direction.UP
            elif event.key == pygame.K_RIGHT and curr_dir != direction.LEFT:
                curr_dir = direction.RIGHT
            elif event.key == pygame.K_DOWN and curr_dir != direction.UP:
                curr_dir = direction.DOWN
            elif event.key == pygame.K_LEFT and curr_dir != direction.RIGHT:
                curr_dir = direction.LEFT
    
    for i in range(len(snake_body)):
        
        if i == 0: snake_body[i][2] = curr_dir
        if snake_body[i][2] == direction.UP:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
            snake_body[i][1] -= UNIT_SIZE 
            pygame.draw.rect(screen, WHITE, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
            
        if snake_body[i][2] == direction.DOWN:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
            snake_body[i][1] += UNIT_SIZE 
            pygame.draw.rect(screen, WHITE, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
            
        if snake_body[i][2] == direction.RIGHT:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
            snake_body[i][0] += UNIT_SIZE 
            pygame.draw.rect(screen, WHITE, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
            
        if snake_body[i][2] == direction.LEFT:
            pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
            snake_body[i][0] -= UNIT_SIZE 
            pygame.draw.rect(screen, WHITE, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))    
    pygame.draw.rect(screen, RED, pygame.Rect(food_position[0], food_position[1], UNIT_SIZE, UNIT_SIZE))
    text_surface = my_font.render(f'Score: {len(snake_body) - 3}', False, (0, 0, 0))
    screen.blit(text_surface, (0,0))
    pygame.display.update()
    
    for i in range(len(snake_body)):
        if i != 0: snake_body[len(snake_body) - i][2] = snake_body[len(snake_body) - i - 1][2]
    clock.tick(10)