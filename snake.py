import pygame
import sys
import numpy as np

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
        
        
SCREEN_SIZE :tuple = (400, 425)
UNIT_SIZE :int = 20
STARTING_POINT :int = 200

# colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

snake_body :list[list] = [SnakeBody(*[STARTING_POINT, STARTING_POINT, direction.UP]),
                          SnakeBody(*[STARTING_POINT, STARTING_POINT + UNIT_SIZE, direction.UP]),
                          SnakeBody(*[STARTING_POINT, STARTING_POINT + 2*UNIT_SIZE, direction.UP])]


curr_dir :direction = direction.UP

pygame.init()
myfont = pygame.font.SysFont("monospace", 16)
score_rect = pygame.Rect(0, 400, 400, 25)

running :bool = True

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

def capture_food():
    global food_position
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

# if snake is out of bounds
def check_window_collision():
    if snake_body[0].x < 0 or snake_body[0].x >= 400 or snake_body[0].y < 0 or snake_body[0].y >= 400:
        global running
        running = False
        pygame.quit()
        sys.exit()

def check_body_collision():
    for i in range(1, len(snake_body)):
        if snake_body[i].x == snake_body[0].x and snake_body[i].y == snake_body[0].y:
            global running
            running = False
            pygame.quit()
            sys.exit()
# game loop
while running:
    capture_food()
    check_window_collision()
    check_body_collision()
    
    already_changed_dir = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and not already_changed_dir:
            if event.key == pygame.K_UP and curr_dir != direction.DOWN:
                curr_dir = direction.UP
                already_changed_dir = True
            elif event.key == pygame.K_RIGHT and curr_dir != direction.LEFT:
                curr_dir = direction.RIGHT
                already_changed_dir = True
            elif event.key == pygame.K_DOWN and curr_dir != direction.UP:
                curr_dir = direction.DOWN
                already_changed_dir = True
            elif event.key == pygame.K_LEFT and curr_dir != direction.RIGHT:
                curr_dir = direction.LEFT
                already_changed_dir = True 
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
    pygame.draw.rect(screen, WHITE, score_rect)
    score = myfont.render("Score: {0}".format(len(snake_body) - 3), 1, BLACK)
    screen.blit(score, (10, 400))
    pygame.draw.rect(screen, RED, pygame.Rect(food_position[0], food_position[1], UNIT_SIZE, UNIT_SIZE))
    
    pygame.display.update()
    
    for i in range(len(snake_body)):
        if i != 0: snake_body[len(snake_body) - i].direction = snake_body[len(snake_body) - i - 1].direction
    clock.tick(20)