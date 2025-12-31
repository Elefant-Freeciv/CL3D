import pygame, math
from cl3d import main
import time, sys
import cProfile, pstats

h = 350*2
w = 525*2
debug = False

'''
for profiling of kernels
cliloader -h -dv python3 main.py

for debugging kernels
export POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1
export POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"
POCL_DEBUG=all gdb python3 main.py
'''
##pygame.init()
##main_screen = pygame.display.set_mode((w, h))
##render_surface = pygame.Surface((w, h))
##font = pygame.font.SysFont("Arial", 15)
##m = main(h, w)
##clock = pygame.time.Clock()
##r = True
### for i in range(1600):
###     m.make()
##profiler = cProfile.Profile()
##profiler.enable()
##while r == True:
##    dt = clock.tick(1000)/1000.0
##    m.update(dt)
##    fps = font.render(str(int(clock.get_fps())), 1, (0, 0, 0))
##    render_surface.fill((255, 255, 255))
##    m.render(render_surface, font)
##    main_screen.blit(render_surface, (0, 0))
##    main_screen.blit(fps, (0, 0))
##    pygame.display.flip()
##    pressed_keys = pygame.key.get_pressed()
##    events = pygame.event.get()
##    for event in events:
##        if event.type == pygame.QUIT:
##            pygame.quit()
##            r = False
##    m.handle_input(events, pressed_keys)
##pygame.quit()
##profiler.disable()
##stats = pstats.Stats(profiler).sort_stats("tottime")
##stats.print_stats()
##sys.exit()
# 
# h = 350
# w = 525
# 
# pygame.init()
# main_screen = pygame.display.set_mode((w, h))
# render_surface = pygame.Surface((w, h))
# font = pygame.font.SysFont("Arial", 15)
# m = main(h, w)
# clock = pygame.time.Clock()
# fast_enough = True
# profiler = cProfile.Profile()
# profiler.enable()
# while fast_enough:
#     for i in range(50):
#         m.make()
#     t=time.time()
#     fpss = []
#     while time.time() < t+3:
#         dt = clock.tick(75)/1000.0
#         m.update(dt)
#         fpss.append(clock.get_fps())
# #         if clock.get_fps() < 58 and time.time() > t+2:
# #             fast_enough = False
# #             pygame.quit()
# #             print("Cube count: ", (len(m.np_points)-9)/36)
# #             sys.exit()
#         fps = font.render(str(int(clock.get_fps())), 1, (0, 0, 0))
#         render_surface.fill((255, 255, 255))
#         m.render(render_surface, font)
#         main_screen.blit(render_surface, (0, 0))
#         main_screen.blit(fps, (0, 0))
#         pygame.display.flip()
#         pressed_keys = pygame.key.get_pressed()
#         events = pygame.event.get()
#         for event in events:
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 fast_enough = False
#         m.handle_input(events, pressed_keys)
#     if sum(fpss)/len(fpss) < 60:
#         fast_enough = False
#         print("Cube count: ", (len(m.np_points)-9)/36)
#   
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("tottime")
# stats.print_stats()
# sys.exit()
            
            
# pygame.init()
# main_screen = pygame.display.set_mode((w, h))
# render_surface = pygame.Surface((w, h))
# font = pygame.font.SysFont("Arial", 15)
# m = main(h, w)
# clock = pygame.time.Clock()
# fast_enough = True
# profiler = cProfile.Profile()
# profiler.enable()
# false = True
# while false:
#     if fast_enough:
#         for i in range(100):
#             m.make()
#     t=time.time()
#     fpss = []
#     while time.time() < t+3:
#         fps_val = int(clock.get_fps())
#         dt = clock.tick(75)/1000.0
#         m.update(dt)
#         fpss.append(fps_val)
# # #         if clock.get_fps() < 58 and time.time() > t+2:
# # #             fast_enough = False
# # #             pygame.quit()
# # #             print("Cube count: ", (len(m.np_points)-9)/36)
# # #             sys.exit()
#         fps = font.render(str(fps_val), 1, (0, 0, 0))
#         render_surface.fill((255, 255, 255))
#         m.render(render_surface, font)
#         main_screen.blit(render_surface, (0, 0))
#         main_screen.blit(fps, (0, 0))
#         pygame.display.flip()
#         pressed_keys = pygame.key.get_pressed()
#         events = pygame.event.get()
#         for event in events:
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 false = False
#         m.handle_input(events, pressed_keys, fps_val)
#     if sum(fpss)/len(fpss) < 60:
#         fast_enough = False
#         print("Cube count: ", (len(m.np_tris)-3)/12)
#   
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("tottime")
# stats.print_stats()
# sys.exit()
#             
#             
# pygame.init()
# main_screen = pygame.display.set_mode((w, h))
# render_surface = pygame.Surface((w, h))
# font = pygame.font.SysFont("Arial", 15)
# m = main(h, w)
# clock = pygame.time.Clock()
# fast_enough = True
# profiler = cProfile.Profile()
# profiler.enable()
# false = True
# m.make()
# while false:
# #     if fast_enough:
# #         for i in range(100):
# #             m.make()
#     t=time.time()
#     fpss = []
#     while time.time() < t+30:
#         fps_val = int(clock.get_fps())
#         dt = clock.tick(75)/1000.0
#         m.update(dt)
#         fpss.append(fps_val)
# # #         if clock.get_fps() < 58 and time.time() > t+2:
# # #             fast_enough = False
# # #             pygame.quit()
# # #             print("Cube count: ", (len(m.np_points)-9)/36)
# # #             sys.exit()
#         fps = font.render(str(fps_val), 1, (0, 0, 0))
#         render_surface.fill((255, 255, 255))
#         m.render(render_surface, font)
#         main_screen.blit(render_surface, (0, 0))
#         main_screen.blit(fps, (0, 0))
#         pygame.display.flip()
#         pressed_keys = pygame.key.get_pressed()
#         events = pygame.event.get()
#         for event in events:
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 false = False
#         m.handle_input(events, pressed_keys, fps_val)
#     if sum(fpss)/len(fpss) < 60:
#         fast_enough = False
#         false = False
#         print("Cube count: ", (len(m.np_tris)-3)/12)
#   
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("tottime")
# stats.print_stats()
# sys.exit()

pygame.init()
main_screen = pygame.display.set_mode((w, h))
render_surface = pygame.Surface((w, h))
font = pygame.font.SysFont("Arial", 15)
m = main(h, w, debug)
clock = pygame.time.Clock()
fast_enough = True
profiler = cProfile.Profile()
profiler.enable()
false = True
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
profiler.disable()
stats = pstats.Stats(profiler).sort_stats("tottime")
stats.print_stats()
sys.exit()
            
            
