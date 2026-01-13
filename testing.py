import pygame, math
from cl3d import main
import time, sys
import cProfile, pstats

h = 350*2
w = 525*2
debug = False

'''
for profiling of kernels
cliloader -h -dv python3 testing.py

for debugging kernels
export POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1
export POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"
POCL_DEBUG=all gdb python3 testing.py
'''

pygame.init()
main_screen = pygame.display.set_mode((w, h))
render_surface = pygame.Surface((w, h))
font = pygame.font.SysFont("Arial", 15)
m = main(h, w, debug)
clock = pygame.time.Clock()
fast_enough = True
# profiler = cProfile.Profile()
# profiler.enable()
false = True
for i in range(5):
    m.make()
for i in range(10):
    print(i)
    fps_val = int(clock.get_fps())
    dt = clock.tick(75)/1000.0
    m.update(dt)
    fps = font.render(str(fps_val), 1, (0, 0, 0))
    render_surface.fill((255, 255, 255))
    m.render(render_surface, font)
    main_screen.blit(render_surface, (0, 0))
    main_screen.blit(fps, (0, 0))
    pygame.display.flip()
    pressed_keys = pygame.key.get_pressed()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            false = False
    m.handle_input(events, pressed_keys, fps_val)
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("tottime")
# stats.print_stats()
pygame.quit()
exit()
sys.exit()
