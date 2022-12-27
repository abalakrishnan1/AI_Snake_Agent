while running:
#     # capture_food()
#     # check_window_collision()
#     # check_body_collision()
    
#     new_state = SnakeBody(snake_body[0].x, snake_body[0].y, snake_body[0].direction)

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#             pygame.quit()
#             sys.exit()
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_UP and curr_dir != direction.DOWN:
#                 snake_body[0].direction = direction.UP
#             elif event.key == pygame.K_RIGHT and curr_dir != direction.LEFT:
#                 snake_body[0].direction = direction.RIGHT
#             elif event.key == pygame.K_DOWN and curr_dir != direction.UP:
#                 snake_body[0].direction = direction.DOWN
#             elif event.key == pygame.K_LEFT and curr_dir != direction.RIGHT:
#                 snake_body[0].direction = direction.LEFT
                
#         if snake_body[0].direction == direction.UP:
#             pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
#             snake_body[0].y -= UNIT_SIZE 
#             pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
            
#         if snake_body[0].direction == direction.DOWN:
#             pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
#             snake_body[i].y += UNIT_SIZE 
#             pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
            
#         if snake_body[i].direction == direction.RIGHT:
#             pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
#             snake_body[i].x += UNIT_SIZE 
#             pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
            
#         if snake_body[i].direction == direction.LEFT:
#             pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
#             snake_body[i].x -= UNIT_SIZE 
#             pygame.draw.rect(screen, GREEN, pygame.Rect(snake_body[0].x, snake_body[0].y, UNIT_SIZE, UNIT_SIZE), 2)
       
#     for i in range(1, len(snake_body)):
#         pygame.draw.rect(screen, BLACK, pygame.Rect(snake_body[i].x, snake_body[i].y, UNIT_SIZE, UNIT_SIZE), 2)
#         old_state = SnakeBody(snake_body[i].x, snake_body[i].y, snake_body[i].direction)
#         pygame.draw.rect(screen, GREEN, pygame.Rect(new_state.x, new_state.y, UNIT_SIZE, UNIT_SIZE), 2)   
#         new_state.x = old_state.x
#         new_state.y = old_state.y
#         new_state.direction = old_state.direction

        
#     pygame.draw.rect(screen, WHITE, score_rect)
#     score = myfont.render("Score: {0}".format(len(snake_body) - 3), 1, BLACK)
#     screen.blit(score, (10, 400))
#     pygame.draw.rect(screen, RED, pygame.Rect(food_position[0], food_position[1], UNIT_SIZE, UNIT_SIZE))
    
#     pygame.display.update()
    
#     clock.tick(5)