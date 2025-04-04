import pyopencl as cl
import pygame, math
import multiprocessing

import numpy as np
from PIL import Image

np.set_printoptions(linewidth=100000, threshold = np.inf)


def sort(points):
    tris = [[]]
    for point in points:
        if len(tris[-1]) < 3:
            tris[-1].append(point)
        else:
            tris[-1].append([len(tris)-1,0,0])
            tris.append([point])
    tris[-1].append([len(tris)-1,0,0])
    tris.sort(
            key=lambda k: sum([vec[2] for vec in k]) / 3, reverse=True)
    trisids = []
    for tri in tris:
        trisids.append(tri[3][0])
    return trisids

class Math3D:
    def translate(mat, vec3):
        mat[0][3] = vec3[0]
        mat[1][3] = vec3[1]
        mat[2][3] = vec3[2]
        
    def rotate(mat, angle, vec3):
        angle_x = vec3[0] * angle
        angle_y = vec3[1] * angle
        angle_z = vec3[2] * angle

        matB = np.eye(4, dtype=np.float32)
        matA = mat

        #x rotation
        matB[1][1] = math.cos(angle_x)
        matB[1][2] = -math.sin(angle_x)
        matB[2][1] = math.sin(angle_x)
        matB[2][2] = math.cos(angle_x)

        matA = np.dot(matA, matB)

        #y rotation
        matB = np.eye(4, dtype=np.float32)
        matB[0][0] = math.cos(angle_y)
        matB[0][2] = math.sin(angle_y)
        matB[2][0] = -math.sin(angle_y)
        matB[2][2] = math.cos(angle_y)

        matA = np.dot(matA, matB)

        #z rotation
        matB = np.eye(4, dtype=np.float32)
        matB[0][0] = math.cos(angle_z)
        matB[0][1] = -math.sin(angle_z)
        matB[1][0] = math.sin(angle_z)
        matB[1][1] = math.cos(angle_z)

        matA = np.dot(matA, matB)
        return matA

class main:
    def __init__(self, h, w, debug=False, target_tile_size=16, slice_size=192):
        self.slice_size = slice_size
        self.debug = debug
        self.h = math.ceil(h/target_tile_size)*target_tile_size
        self.w = math.ceil(w/target_tile_size)*target_tile_size
        self.y = int(self.h / target_tile_size)
        self.x = int(self.w / target_tile_size)
        self.tilesizex = target_tile_size
        self.tilesizey = target_tile_size
        self.pre_dims = (int(self.y/4),int(self.x/6))
        print(self.pre_dims)
        self.viewpos = [0.0, 0.0, -10.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.delta = 0.0
        self.clicking = False
        self.start_click = []
#         self.ctx = cl.Context(dev_type=cl.device_type.CPU,
#             properties=[(cl.context_properties.PLATFORM, cl.get_platforms()[1])])
        self.ctx = cl.Context(dev_type=cl.device_type.GPU,
            properties=[(cl.context_properties.PLATFORM, cl.get_platforms()[0])])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx,
        f'''
        #define slice_size {self.slice_size}
        #define tilesize (uint2)({self.tilesizey}, {self.tilesizex})
        #define tilecount (uint2)({self.y}, {self.x})
        #define pre_dims (uint2)({self.pre_dims[0]},{self.pre_dims[1]})
        #define pre_scale (uint2)(4,6)
        #define XCOUNT {self.tilesizey+1}
        #define YCOUNT {self.tilesizex+1}

        typedef int tile_layer[{self.y}][{self.x}];
        typedef uchar bool_layer[{self.y}][{self.x}];
        typedef uchar pre_layer[{self.pre_dims[0]}][{self.pre_dims[1]}];
        typedef int preint_layer[{self.pre_dims[0]}][{self.pre_dims[1]}];
        typedef uchar4 scr_img[{self.h}][{self.w}];
        typedef uchar4 tex_img[256][1024][1024];
        '''+open("kernels.cl").read()).build()#options=["-cl-fast-relaxed-math","-cl-nv-verbose"])
        
        mf = cl.mem_flags

        vertices = [(10.0, 0.0, 0.0),  #x axis
                    (-10.0, 0.0, 0.0), #x axis
                    (10.0, 0.0, .1),  #x axis
                    (0.0, 10.0, .1),  #y axis
                    (0.0, 10.0, 0.0),  #y axis
                    (0.0, -10.0, 0.0),  #y axis
                    (.1, 0.0, 10.0),  #z axis
                    (0.0, 0.0, 10.0),  #z axis
                    (0.0, 0.0, -10.0)]
        triangles = [(0,1,2,0),#x axis
                (3,4,5,0),#y axis
                (6,7,8,0)]#z axis
        tex_coords = [(0, 0, 255, 0, 0, 255,1,1),
                    (0, 0, 255, 0, 0, 255,1,1),
                    (0, 0, 255, 0, 0, 255,1,1)]
        colours = [(0,0,0,255),
                   (0,0,0,255),
                   (0,0,0,255)]
        self.np_colours = np.array(colours, dtype=cl.cltypes.uchar)
        print(self.np_colours)
        self.cl_colours = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_colours)
        points = []
        for v in vertices:
            points.append((v[0], v[1], v[2], 1.0))
        self.np_points = np.array(points, dtype=np.float32)
        tris = []
        for tri in triangles:
            tris.append((tri[0], tri[1], tri[2], 1))
        self.np_tris = np.array(tris, dtype=cl.cltypes.uint)
        print(self.np_tris)
        self.cl_tris = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_tris)

        self.texc = []
        for coord in tex_coords:
            self.texc.append(coord)
        self.np_tex_coords = np.array(self.texc, dtype=np.float32)
        print(self.np_tex_coords)
        self.tex_coords = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_tex_coords)
        
        view = [[0.08333333333333333, 0.0, 0.0, 0.0], [0.0, 0.125, 0.0, 0.0], [0.0, 0.0, -0.02002002002002002, -0.8018018018018018], [0, 0, 0, 1.0]]
        np_view = np.array(view, dtype=np.float32)
        print(np_view)

        model = [[1.0, 0.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0, 1.131144640],
                 [0.0, 0.0, 1.0, 1.0],
                 [0.0, 0.0, 0.0, 1.0]]
        np_model = np.array(model, dtype=np.float32)
        self.src_img1 = Image.open('plane.jpg').convert('RGBA')
        self.src_img2 = Image.open('plane.jpg').convert('RGBA')
        self.src = np.zeros((2,1024,1024,4), dtype=cl.cltypes.uchar)

        self.src[0] = np.array(self.src_img1, dtype=cl.cltypes.uchar)
        self.src[1] = np.array(self.src_img2, dtype=cl.cltypes.uchar)
        self.tex = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.src)
        
        np_screen = np.array([[self.w, self.h]], dtype=np.float32)

        self.cl_screen = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_screen)
        self.cl_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.np_points.nbytes)
        print(len(vertices))
        
        self.vertex_shader = self.prg.vertex
        self.make_tiles1 = self.prg.make_tiles1
        self.make_tiles2 = self.prg.make_tiles2
        self.count_tiles = self.prg.count_tiles
        self.tiles1 = self.prg.make_tiles_stage_1
        self.tiles2 = self.prg.make_tiles_stage_2
        self.tiles3 = self.prg.make_tiles_stage_3
        self.tiles4 = self.prg.make_tiles_stage_4
        
        
    def update(self, delta):
        self.delta = delta
        if self.clicking:
            rel_pos = [pygame.mouse.get_pos()[1] - self.start_click[1], pygame.mouse.get_pos()[0] - self.start_click[0]]
            self.rotation[0] = -rel_pos[0] / self.w
            self.rotation[1] = -rel_pos[1] / self.h
            
    def handle_input(self, events, pressed_keys, fps):
        mod = 60/(fps+1)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.clicking == False:
                    self.start_click = [pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]]
                    self.clicking = True
                if event.button == 3:
                    self.make()
            if event.type == pygame.MOUSEBUTTONUP:
                self.clicking = False
                self.start_click = [pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]]
        if pressed_keys[pygame.K_UP]:
            self.viewpos[1] += 0.1*mod
        if pressed_keys[pygame.K_DOWN]:
            self.viewpos[1] -= 0.1*mod
        if pressed_keys[pygame.K_LEFT]:
            self.viewpos[0] -= 0.1*mod
        if pressed_keys[pygame.K_RIGHT]:
            self.viewpos[0] += 0.1*mod
        if pressed_keys[pygame.K_PAGEUP]:
            self.viewpos[2] -= 0.1*mod
        if pressed_keys[pygame.K_PAGEDOWN]:
            self.viewpos[2] += 0.1*mod
        if pressed_keys[pygame.K_e]:
            self.rotation[1] += 0.01*mod
        if pressed_keys[pygame.K_q]:
            self.rotation[1] -= 0.01*mod
        if pressed_keys[pygame.K_w]:
            self.rotation[0] += 0.0131144641*mod
        if pressed_keys[pygame.K_s]:
            self.rotation[0] -= 0.01*mod
            
    def make(self):
        vertices = []
        file = open("plane.obj").read().splitlines()
        for line in file:
            if line.startswith("v "):
                l = line.split()
                vertices.append((float(l[1]),float(l[2]),float(l[3])))
        
        tex_coords_u = []
        for line in file:
            if line.startswith("vt "):
                l = line.split()
                a = float(l[1])
                b = float(l[2])
                a = a*1023
                b = b*1023
                tex_coords_u.append((b,a))
               
        tex_coords = []
        triangles = []
        for line in file:
            if line.startswith("f "):
                l = line.split()
                triangles.append(((int(l[1].split("/")[0])-1), int(l[2].split("/")[0])-1, int(l[3].split("/")[0])-1))
                a=tex_coords_u[int(l[1].split("/")[1])-1]
                b=tex_coords_u[int(l[2].split("/")[1])-1]
                c=tex_coords_u[int(l[3].split("/")[1])-1]
                tex_coords.append((1024-a[0], a[1], 1024-b[0], b[1], 1024-c[0], c[1], 1, 0))
        mf = cl.mem_flags
        
        tris = []
        tcount = self.np_points.shape[0]
        for tri in self.np_tris:
            tris.append((tri[0], tri[1], tri[2], 1))
        for tri in triangles:
            tris.append((tri[0]+tcount, tri[1]+tcount, tri[2]+tcount, 1))

        self.np_tris = np.array(tris, dtype=cl.cltypes.uint)

        self.cl_tris = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_tris)
        
        for coord in tex_coords:
            self.texc.append(coord)
        self.np_tex_coords = np.array(self.texc, dtype=np.float32)
        self.tex_coords = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_tex_coords)
        
        points = []
        print(self.np_points.nbytes)
        for vert in self.np_points:
            points.append((vert[0], vert[1], vert[2], 1.0))
        for v in vertices:
            points.append((v[0], v[1], v[2], 1.0))
        self.np_points = np.array(points, dtype=np.float32)
        c = [(255, 100, 100, 255),
                   (255, 100, 100, 255),
                   (255, 255, 100, 255),
                   (255, 255, 100, 255),
                   (255, 100, 255, 255),
                   (255, 100, 255, 255),
                   (100, 255, 100, 255),
                   (100, 255, 100, 255),
                   (100, 255, 255, 255),
                   (100, 255, 255, 255),
                   (100, 0, 100, 255),
                   (100, 0, 100, 255)]
        colours = []
        for colour in self.np_colours:
            colours.append(colour)
        for colour in c:
            colours.append(colour)
        self.np_colours = np.array(colours, dtype=cl.cltypes.uchar)
        self.cl_colours = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_colours)

    def rgba2rgb(self, rgba, background=(0,0,0)):
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros( (row, col, 3), dtype='float32' )
        r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

        a = np.asarray( a, dtype='float32' ) / 255.0

        R, G, B = background

        rgb[:,:,0] = r * a + (1.0 - a) * R
        rgb[:,:,1] = g * a + (1.0 - a) * G
        rgb[:,:,2] = b * a + (1.0 - a) * B

        return np.asarray( rgb, dtype='uint8' )
        
    def render(self, render_surface, font):
        debug = self.debug
        if debug:
            step = 0
            print("step: ", step)#0
            step += 1
        np_model = np.eye(4, dtype=np.float32)
        Math3D.translate(np_model, (1.0, 1.0, 1.0))
        np_model = Math3D.rotate(np_model, 10.0 * self.rotation[0], (1.0, 0.0, 0.0))
        np_model = Math3D.rotate(np_model, 10.0 * self.rotation[1], (0.0, 1.0, 0.0))
        
        if debug:
            print("step: ", step)#1
            step += 1
        
        view = np.eye(4, dtype=np.float32)
        Math3D.translate(view, (-self.viewpos[0], -self.viewpos[1], self.viewpos[2]))
        
        if debug:
            print("step: ", step)#2
            step += 1
        
        right = 12.0
        top = 8.0
        far = 100.0
        near = 0.1
        
        if debug:
            print("step: ", step)#3
            step += 1
        
        orth_proj = np.eye(4, dtype=np.float32)
        orth_proj[0][0] = 1 / right
        orth_proj[1][1] = 1 / top
        orth_proj[2][2] = -2 / (far - near)
        orth_proj[2][3] = -((far + near) / (far - near))
        
        if debug:
            print("step: ", step)#4
            step += 1
        
        np_view = np.dot(orth_proj, view)
        np_view = np.dot(np_view, np_model)
        
        mf = cl.mem_flags
        
        self.cl_points = cl.Buffer(self.ctx,
                                   mf.READ_WRITE | mf.COPY_HOST_PTR,
                                   hostbuf=self.np_points)
        
        self.cl_view = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=np_view)
        
        self.cl_model = cl.Buffer(self.ctx,
                                  mf.READ_ONLY | mf.COPY_HOST_PTR,
                                  hostbuf=np_model)
        
        self.cl_out = cl.Buffer(self.ctx,
                                mf.READ_WRITE,
                                self.np_points.nbytes)
        
        self.vertex_shader(self.queue,
                           (self.np_points.shape[0],),
                           None,
                           self.cl_points,
                           self.cl_view,
                           self.cl_screen,
                           self.cl_out).wait()

        slice_count = math.ceil(self.np_tris.shape[0]/self.slice_size)
        
        if debug:
            print("slice_count: ", slice_count)
        
        self.mapsize = int(self.np_tris.shape[0])
        
        self.cl_tile_maps = cl.Buffer(self.ctx,
                                      mf.READ_WRITE,
                                      (self.y*self.x*self.mapsize))
        
        cl.enqueue_fill_buffer(self.queue,
                               self.cl_tile_maps,
                               cl.cltypes.uchar(0),
                               0,
                               (self.y*self.x*self.mapsize))
        
        self.cl_tile_premaps = cl.Buffer(self.ctx,
                                         mf.READ_WRITE,
                                         (int(self.y*self.x*self.mapsize/24)))
        
        self.cl_tile_count = cl.Buffer(self.ctx,
                                       mf.READ_WRITE,
                                       (4*self.y*self.x*slice_count))
        
        self.cl_tile_prelayer = cl.Buffer(self.ctx,
                                          mf.READ_WRITE,
                                          int(4*self.y*self.x/24)*slice_count)
        
        if debug:
            print("step: ", step)#5
            step += 1
        
        self.tiles1(self.queue,
                    (self.mapsize, self.pre_dims[0], self.pre_dims[1]),
                    None,
                    self.cl_tris,
                    self.cl_out,
                    self.cl_tile_premaps).wait()
        
        if debug:
            print("step: ", step)#6
            step += 1
        
        self.tiles2(self.queue,
                    (self.pre_dims[0], self.pre_dims[1], slice_count),
                    None,
                    self.cl_tile_premaps,
                    self.cl_tile_prelayer,
                    cl.cltypes.uint(self.np_tris.shape[0])).wait()
        
        if debug:
            print("step: ", step)#7
            step += 1
        
        np_tile_prelayer = np.empty((slice_count, self.pre_dims[0], self.pre_dims[1]), dtype=np.int32)
        cl.enqueue_copy(self.queue, np_tile_prelayer, self.cl_tile_prelayer)
        np_out2 = np.sum(np_tile_prelayer, axis=0)
        
        if debug:
            print("step: ", step)#8
            step += 1
        
        self.cl_sorted_tris = cl.Buffer(self.ctx,
                                        mf.READ_WRITE,
                                        (4*np.sum(np_out2)))
        
        if debug:
            print("step: ", step)#9
            step += 1
        
        self.np_offsets = np.empty((self.pre_dims[0], self.pre_dims[1]), dtype=np.int32)
        r_offset=0
        for i in range(self.pre_dims[0]):
            for j in range(self.pre_dims[1]):
                self.np_offsets[i][j]=r_offset
                r_offset += np_out2[i][j]
        self.cl_offsets = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=self.np_offsets)
        
        if debug:
            print("step: ", step)#10
            step += 1
        
        self.tiles3(self.queue,
                    (self.pre_dims[0], self.pre_dims[1], slice_count),
                    None,
                    self.cl_sorted_tris,
                    self.cl_tile_premaps,
                    self.cl_offsets,
                    self.cl_tile_prelayer,
                    cl.cltypes.uint(self.np_tris.shape[0])).wait()
        
        if debug:
            print("step: ", step)#11
            step += 1
        
        np_out_l = np.empty((np.sum(np_out2)), dtype=np.int32)
        cl.enqueue_copy(self.queue, np_out_l, self.cl_sorted_tris)
        
        if debug:
            print("step: ", step)#12
            step += 1

        self.tiles4(self.queue,
                    (np.sum(np_out2), 4, 6),
                    (1, 4, 6),
                    self.cl_sorted_tris,
                    self.cl_offsets,
                    self.cl_tris,
                    self.cl_out,
                    self.cl_tile_maps).wait()
        
        if debug:
            print("step: ", step)#13
            step += 1
            np_tile_maps = np.empty((self.mapsize, self.y, self.x), dtype=cl.cltypes.uchar)
            cl.enqueue_copy(self.queue, np_tile_maps, self.cl_tile_maps)
            print(np_tile_maps)
            
        #self.make_tiles1(self.queue, (self.mapsize,), None, self.cl_tris, self.cl_out, self.cl_tile_maps)#, self.cl_tile_layers)
        #self.prg.old_make_tiles1(self.queue, (self.mapsize, self.y, self.x), None, self.cl_tris, self.cl_out, self.cl_tile_maps)
        
        self.count_tiles(self.queue,
                         (self.y,self.x, slice_count),
                         None,
                         self.cl_tile_maps,
                         self.cl_tile_count,
                         cl.cltypes.uint(self.np_tris.shape[0])).wait()
        
        if debug:
            print("step: ", step)#14
            step += 1
        
        np_out = np.empty((slice_count, self.y, self.x), dtype=np.int32)
        cl.enqueue_copy(self.queue, np_out, self.cl_tile_count)
        self.cl_tile_layers = cl.Buffer(self.ctx,
                                        mf.READ_WRITE,
                                        max(4*self.y*self.x*np.sum(np_out, axis=0).max(), 4*self.y*self.x))

        if debug:
            print("step: ", step)#15
            step += 1
        
        np_tile_layer = np.sum(np_out, axis=0, dtype=np.int32)
        
        if debug:
            print("np_out: ", np_out)
            print("cl_tile_layer: ", np_tile_layer)
        
        self.cl_tile_layer = cl.Buffer(self.ctx,
                                       mf.READ_ONLY|mf.COPY_HOST_PTR,
                                       hostbuf=np_tile_layer)
        
        if debug:
            print("step: ", step)#16
            step += 1
        
        self.make_tiles2(self.queue,
                         (self.y,self.x, slice_count),
                         None,
                         self.cl_tile_maps,
                         self.cl_tile_layers,
                         self.cl_tile_count,
                         cl.cltypes.uint(self.np_tris.shape[0])).wait()
        
        if debug:
            print("step: ", step)#17
            step += 1

        self.dest = np.empty((self.h,self.w,4), dtype=cl.cltypes.uchar)
        self.dest_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dest)
        self.prg.draw_tris(self.queue,
                           (self.h, self.w),
                           (self.tilesizex, self.tilesizey),
                           self.cl_tris,
                           self.cl_out,
                           self.tex_coords,
                           self.cl_colours,
                           self.cl_tile_layers,
                           self.cl_tile_layer,
                           self.tex,
                           self.dest_buf).wait()
        
        if debug:
            print("step: ", step)#18
            step += 1
        
        cl.enqueue_copy(self.queue, self.dest, self.dest_buf)
        
        if debug:
            print("step: ", step)#19
            step += 1

        surf = pygame.surfarray.make_surface(self.dest[:,:,:3])
        surf = pygame.transform.rotate(surf, 90)
        surf = pygame.transform.flip(surf, False, True)
        render_surface.blit(surf, (0, 0))
        verts = font.render(str(len(self.np_points)), 1, (0, 0, 0))
        render_surface.blit(verts, (0, 30))
        
        if debug:
            print("step: ", step)#20
            step += 1
        
#         self.cl_tile_layers.release()
#         self.cl_tile_layer.release()
#         self.cl_tile_maps.release()
#         self.cl_out.release()
#         self.cl_model.release()
#         self.cl_view.release()
#         self.cl_points.release()
#         self.dest_buf.release()
#         self.cl_tile_count.release()
        
        if debug:
            print("step: ", step)#21
            step += 1
            for i in range(self.y):
                for j in range(self.x):
                    if np_tile_layer[i][j] < 100:
                        render_surface.blit(font.render(str(np_tile_layer[i][j]), 1, (0, 0, 0)), (j*self.tilesizex, i*self.tilesizey))
                    else:
                        render_surface.blit(font.render("XX", 1, (0, 0, 0)), (j*self.tilesizex, i*self.tilesizey))
#         for i in range(self.pre_dims[0]):
#             for j in range(self.pre_dims[1]):
#                 render_surface.blit(font.render(str(np_out2[i][j]), 1, (0, 0, 0)), (j*self.tilesizex*6, i*self.tilesizey*4))




