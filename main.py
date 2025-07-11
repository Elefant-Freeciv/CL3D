import pygame, math
from cl3d import main

h = 350*2
w = 525*2
target_tile_size = 16
debug = False

pygame.init()
main_screen = pygame.display.set_mode((w, h))
render_surface = pygame.Surface((w, h))
font = pygame.font.SysFont("Arial", 15)
m = main(h, w, debug, target_tile_size)
clock = pygame.time.Clock()
r = True
while r == True:
    dt = clock.tick(1000)/1000.0
    m.update(dt)
    fps_val = int(clock.get_fps())
    fps = font.render(str(fps_val), 1, (0, 0, 0))
    render_surface.fill((255, 255, 255))
    m.render(render_surface, font)
    main_screen.blit(render_surface, (0, 0))
    main_screen.blit(fps, (0, 15))
    pygame.display.flip()
    pressed_keys = pygame.key.get_pressed()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            r = False
    m.handle_input(events, pressed_keys, fps_val)
pygame.quit()