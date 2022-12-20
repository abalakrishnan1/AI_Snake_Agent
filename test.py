import pygame, time

square_color = (255, 255, 255)

pygame.init()

screen = pygame.display.set_mode((400, 400))

pygame.draw.rect(screen, square_color, pygame.Rect(200, 200, 20, 20))

pygame.display.flip()

i = 1

while pygame.event.get != pygame.K_ESCAPE:
    pygame.draw.rect(screen, square_color, pygame.Rect(200, 200 + i, 20, 20))
    pygame.display.flip()
    time.sleep(0.1)
    i += 1
