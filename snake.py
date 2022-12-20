import pygame, sys, numpy as np
from enum import Enum

# constants
class direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4   

SCREEN_SIZE = (400, 400)
UNIT_SIZE = 20

# colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)


snake_body = [[200, 200, direction.UP],
              [200, 220, direction.UP],
              [200, 240, direction.UP]]

curr_dir = direction.UP

pygame.init()

screen = pygame.display.set_mode(SCREEN_SIZE)

# init render
for i in range(len(snake_body)):
    pygame.draw.rect(screen, WHITE, pygame.Rect(snake_body[i][0], snake_body[i][1], UNIT_SIZE, UNIT_SIZE))
pygame.display.update()

clock = pygame.time.Clock()

food_position = [np.random.randint(), np.random.randint()]

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                curr_dir = direction.UP
            elif event.key == pygame.K_RIGHT:
                curr_dir = direction.RIGHT
            elif event.key == pygame.K_DOWN:
                curr_dir = direction.DOWN
            elif event.key == pygame.K_LEFT:
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

    pygame.display.update()
    
    for i in range(len(snake_body)):
        if i != 0: snake_body[len(snake_body) - i][2] = snake_body[len(snake_body) - i - 1][2]
    clock.tick(10)