import pyopencl as cl
import pygame, math
import multiprocessing

import numpy as np
from multiprocessing import Process
from PIL import Image

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
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.viewpos = [0.0, 0.0, -10.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.delta = 0.0
        self.clicking = False
        self.start_click = []
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, '''//CL//

        typedef int tile_layer[10][7];
        
        float4 mul(__global const float4 mat[4], const float4 point)
        {
            float4 rtn;
            rtn.x = dot(mat[0], point);
            rtn.y = dot(mat[1], point);
            rtn.z = dot(mat[2], point);
            rtn.w = dot(mat[3], point);
            return rtn;

        }
        
        bool orient(float4 A,float4 B,float2 C)
        {
            float2 AB = (float2)(B.x-A.x, B.y-A.y);
            float2 AC = (float2)(C.x-A.x, C.y-A.y);
            float cross = (AB.x * AC.y) - (AB.y * AC.x);
            if (cross > 0){return true;}
            else{return false;}
        }

        bool point_in_triangle (int2 pos, float4 v1, float4 v2, float4 v3)
        {
            float2 pt = (float2)(convert_float(pos.x),convert_float(pos.y));
            return ((orient(v1, v2, pt) && orient(v2, v3, pt) && orient(v3, v1, pt))||(!orient(v1, v2, pt) && !orient(v2, v3, pt) && !orient(v3, v1, pt)));
        }
        
        bool lines_intersect(float4 p1, float4 p2, int4 tilerect)
        {
            float2 p3 = (float2)(convert_float(tilerect.x), convert_float(tilerect.y));
            float2 p4 = (float2)(convert_float(tilerect.z), convert_float(tilerect.w));
            float d = p4.x*p3.y - p3.x*p4.y;
            float s = (1/d)*((p1.x - p2.x)*p3.y - (p1.y - p2.y)*p3.x);
            float t = (1/d)*(-(-(p1.x - p2.x)*p4.y + (p1.y - p2.y)*p4.x));
            if (s <= 1 && s >= 0 && t <= 1 && t >= 0){return true;}
            else {return false;}
        }
        
        float my_distance(float4 p1, float4 p2)
        {
            return sqrt(pow((p2.x-p1.x), 2)+pow((p2.y-p1.y), 2));
        }
        
        float3 barycentric(float2 px, float4 a, float4 b, float4 c)
        {
            float u, v, w;
            float4 p = (float4)(px.x, px.y, 0, 0);
            float4 v0 = b - a, v1 = c - a, v2 = p - a;
            float d00 = dot(v0, v0);
            float d01 = dot(v0, v1);
            float d11 = dot(v1, v1);
            float d20 = dot(v2, v0);
            float d21 = dot(v2, v1);
            float denom = d00 * d11 - d01 * d01;
            v = (d11 * d20 - d01 * d21) / denom;
            w = (d00 * d21 - d01 * d20) / denom;
            u = 1.0f - v - w;
            return (float3)(u, v, w);
        }
        
        float pixel_depth(int2 pos, float4 p1, float4 p2, float4 p3)
        {
            float2 px = (float2)(convert_float(pos.x),convert_float(pos.y));
            float3 bary = barycentric(px, p1, p2, p3);
            float z = bary.x * 1/p1.w + bary.y * 1/p2.w + bary.z * 1/p3.w;
            return 1/z;
        }
        
        uint4 texture_pixel(int2 pos, int i, float z, read_only image2d_t tex, __global float4 *tex_coords, __global float4 *tris)
        {
            float4 p1 = tris[i];
            float4 p2 = tris[i+1];
            float4 p3 = tris[i+2];
            
            float2 px = (float2)(convert_float(pos.x),convert_float(pos.y));
            float3 bary = barycentric(px, p1, p2, p3);
            
            float2 st0 = (float2)(tex_coords[i].x, tex_coords[i].y);
            float2 st1 = (float2)(tex_coords[i+1].x, tex_coords[i+1].y);
            float2 st2 = (float2)(tex_coords[i+2].x, tex_coords[i+2].y);
            
            st0[0] /= p1.w, st0[1] /= p1.w;
            st1[0] /= p2.w, st1[1] /= p2.w;
            st2[0] /= p3.w, st2[1] /= p3.w;
            
            float x = bary.x * st0[0] + bary.y * st1[0] + bary.z * st2[0];
            float y = bary.x * st0[1] + bary.y * st1[1] + bary.z * st2[1];
            x *= z, y *= z;
            
            const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
            return read_imageui(tex, sampler, (float2)(x, y));
        }
        
        __kernel void vertex(__global const float4 *points,
                             __global const float4 mat[4],
                             __global const float2 *screen,
                             __global float4 *out)
        {
            float4 workvec;
            int gid = get_global_id(0);
            //run custom vertex code
            workvec = mul(mat, points[gid]);
            workvec.w = -workvec.z;
            workvec = (float4)(workvec.x / workvec.w, workvec.y / workvec.w, workvec.z / workvec.w, -workvec.z);
            float x = ((workvec.x + 1) / 2) * screen[0].x;
            float y = ((-workvec.y + 1) / 2) * screen[0].y;
            out[gid] = (float4)(y, x, 0, workvec.w);
        }
        
        __kernel void make_tiles1(
                                    __global const float4 *tris,
                                    __global tile_layer *bool_map,
                                    __global tile_layer *tile_layers,
                                    uint tilesizex,
                                    uint tilesizey
                                 )
        {
            int tri = get_global_id(0)*3;
            const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
            float4 p1 = tris[tri];
            float4 p2 = tris[tri+1];
            float4 p3 = tris[tri+2];
            uint2 tilesize = (uint2)(tilesizex, tilesizey);
            int2 tile = (int2)(get_global_id(1), get_global_id(2));
            //printf("[%i|%i|%i|%i|%i]", tile.x, tile.y, tilesize.x, tilesize.y, tri);
            bool_map[tri/3][tile.x][tile.y] = 0;
            tile_layers[tri/3][tile.x][tile.y] = 0;
            int4 tilerect = (int4)(tile.x*tilesize.x, tile.y*tilesize.y, tile.x*tilesize.x+tilesize.x, tile.y*tilesize.y+tilesize.y);
            bool a = (p1.x >= tilerect.x && p1.x <= tilerect.z);
            bool b = (p2.x >= tilerect.x && p2.x <= tilerect.z);
            bool c = (p1.y >= tilerect.y && p1.y <= tilerect.w);
            bool d = (p2.y >= tilerect.y && p2.y <= tilerect.w);
            bool e = (p3.x >= tilerect.x && p3.x <= tilerect.z);
            bool f = (p3.y >= tilerect.y && p3.y <= tilerect.w);
            if ((a && c) || (b && d) || (e && f)){bool_map[tri/3][tile.x][tile.y] = 1;}
            a = point_in_triangle((int2)(tilerect.x, tilerect.y), p1, p2, p3);
            b = point_in_triangle((int2)(tilerect.x, tilerect.w), p1, p2, p3);
            c = point_in_triangle((int2)(tilerect.z, tilerect.y), p1, p2, p3);
            d = point_in_triangle((int2)(tilerect.z, tilerect.w), p1, p2, p3);
            if (a || b || c || d){bool_map[tri/3][tile.x][tile.y] = 1;}
            a = lines_intersect(p1, p2, tilerect);
            b = lines_intersect(p2, p3, tilerect);
            c = lines_intersect(p3, p1, tilerect);
            if (a || b || c){bool_map[tri/3][tile.x][tile.y] = 1;}
        }
        
        __kernel void make_tiles2(__global tile_layer *bool_map, __global tile_layer *out, __global tile_layer tri_count, uint pcount)
        {
            int2 tile = (int2)(get_global_id(0), get_global_id(1));
            tri_count[tile.x][tile.y]=0;
            int j = 0;
            for (int i = 0; i<pcount; i++)
            {
                if (bool_map[i][tile.x][tile.y]==1)
                {
                    out[j][tile.x][tile.y]=i;
                    j++;
                    tri_count[tile.x][tile.y]=j;
                }
                else {out[j][tile.x][tile.y]=0;}
            }
        }
        
        __kernel void draw_tris(
            __global const float4 *tris,
            __global const float4 *tex_coords,
            uint pcount,
            __global const uint4 *colours,
            __global tile_layer *tile_maps,
            __global tile_layer tri_count,
            read_only image2d_t tex,
            write_only image2d_t screen,
            uint tilesizex,
            uint tilesizey)
            {
                const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
                int2 tile = (int2)(get_global_id(0)/tilesizey, get_global_id(1)/tilesizex);
                int2 pos = (int2)(get_global_id(0), get_global_id(1));
                write_imageui(screen, pos, (uint4)(tile.x*25,tile.y*25,255,255));//(uint4)(pos.x,pos.y,convert_int(tris[0].x),255));
                float old_pixel_depth = 100000;
                float test_pixel_depth;
                for (int i = 0; i<(tri_count[tile.y][tile.x]*3); i += 3)
                {
                    
                    if(point_in_triangle(pos, tris[tile_maps[i/3][tile.y][tile.x]*3], tris[tile_maps[i/3][tile.y][tile.x]*3+1], tris[tile_maps[i/3][tile.y][tile.x]*3+2]))
                    {
                        test_pixel_depth = pixel_depth(pos, tris[tile_maps[i/3][tile.y][tile.x]*3], tris[tile_maps[i/3][tile.y][tile.x]*3+1], tris[tile_maps[i/3][tile.y][tile.x]*3+2]);
                        if(test_pixel_depth < old_pixel_depth)
                        {
                            uint4 colour = texture_pixel(pos, tile_maps[i/3][tile.y][tile.x]*3, test_pixel_depth, tex, tex_coords, tris);
                            // custom fragment shader here
                            //colour /= (convert_uint(test_pixel_depth*10));
                            //printf("[%f]", test_pixel_depth);
                            write_imageui(screen, pos, colour);
                            old_pixel_depth = test_pixel_depth;
                        }
                    }
                }
            }
        ''').build()
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
        tex_coords = [(0, 0),
                    (255, 0),
                    (0, 255),
                    (0, 0),
                    (0, 255),
                    (255, 0),
                    (255, 0),
                    (0, 0),
                    (0, 255)]
        colours = [(255,0,0,255),
                   (0,255,0,255),
                   (0,0,255,255)]
        self.np_colours = np.array(colours, dtype="uint")
        self.cl_colours = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_colours)
        points = []
        for v in vertices:
            points.append((v[0], v[1], v[2], 1.0))
        self.np_points = np.array(points, dtype=np.float32)
        print(self.np_points)
        self.texc = []
        for coord in tex_coords:
            self.texc.append([coord[0], coord[1],0,0])
        np_texc = np.array(self.texc, dtype=np.float32)
        print(np_texc)
        self.tex_coords = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_texc)
        
        view = [[0.08333333333333333, 0.0, 0.0, 0.0], [0.0, 0.125, 0.0, 0.0], [0.0, 0.0, -0.02002002002002002, -0.8018018018018018], [0, 0, 0, 1.0]]
        np_view = np.array(view, dtype=np.float32)
        print(np_view)

        model = [[1.0, 0.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [0.0, 0.0, 1.0, 1.0],
                 [0.0, 0.0, 0.0, 1.0]]
        np_model = np.array(model, dtype=np.float32)
        print(np_model)
        self.src_img = Image.open('Tex2.png').convert('RGBA')
        self.src = np.array(self.src_img)
        self.tex = cl.image_from_array(self.ctx, self.src, 4)
        
        np_screen = np.array([[self.w, self.h]], dtype=np.float32)

        self.cl_screen = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_screen)
        self.cl_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.np_points.nbytes)
        print(len(vertices))
        
        self.vertex_shader = self.prg.vertex
        self.make_tiles1 = self.prg.make_tiles1
        self.make_tiles2 = self.prg.make_tiles2
        
        
    def update(self, delta):
        self.delta = delta
        if self.clicking:
            rel_pos = [pygame.mouse.get_pos()[1] - self.start_click[1], pygame.mouse.get_pos()[0] - self.start_click[0]]
            self.rotation[0] = -rel_pos[0] / self.w
            self.rotation[1] = -rel_pos[1] / self.h
            
    def handle_input(self, events, pressed_keys):
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
            self.viewpos[1] += 0.1
        if pressed_keys[pygame.K_DOWN]:
            self.viewpos[1] -= 0.1
        if pressed_keys[pygame.K_LEFT]:
            self.viewpos[0] -= 0.1
        if pressed_keys[pygame.K_RIGHT]:
            self.viewpos[0] += 0.1
        if pressed_keys[pygame.K_PAGEUP]:
            self.viewpos[2] -= 0.1
        if pressed_keys[pygame.K_PAGEDOWN]:
            self.viewpos[2] += 0.1
        if pressed_keys[pygame.K_e]:
            self.rotation[1] += 0.01
        if pressed_keys[pygame.K_q]:
            self.rotation[1] -= 0.01
        if pressed_keys[pygame.K_w]:
            self.rotation[0] += 0.01
        if pressed_keys[pygame.K_s]:
            self.rotation[0] -= 0.01
            
    def make(self):
        vertices=[[-1.0, -1.0, -1.0],
                 [1.0, -1.0, -1.0],
                 [1.0, 1.0, -1.0],
                 [1.0, 1.0, -1.0],
                 [-1.0, 1.0, -1.0],
                 [-1.0, -1.0, -1.0],

                 [-1.0, -1.0, 1.0],
                 [1.0, -1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [-1.0, 1.0, 1.0],
                 [-1.0, -1.0, 1.0],

                 [-1.0, 1.0, 1.0],
                 [-1.0, 1.0, -1.0],
                 [-1.0, -1.0, -1.0],
                 [-1.0, -1.0, -1.0],
                 [-1.0, -1.0, 1.0],
                 [-1.0, 1.0, 1.0],

                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, -1.0],
                 [1.0, -1.0, -1.0],
                 [1.0, -1.0, -1.0],
                 [1.0, -1.0, 1.0],
                 [1.0, 1.0, 1.0],

                 [-1.0, -1.0, -1.0],
                 [1.0, -1.0, -1.0],
                 [1.0, -1.0, 1.0],
                 [1.0, -1.0, 1.0],
                 [-1.0, -1.0, 1.0],
                 [-1.0, -1.0, -1.0],

                 [-1.0, 1.0, -1.0],
                 [1.0, 1.0, -1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [-1.0, 1.0, 1.0],
                 [-1.0, 1.0, -1.0]]
        tex_coords = [(255, 0),
                    (0, 0),
                    (0, 255),
                    (0, 255),
                    (0, 0),
                    (255, 0),
                      
                    (255, 0),
                    (0, 0),
                    (0, 255),
                    (0, 255),
                    (255, 255),
                    (255, 0),
                      
                    (255, 0),
                    (0, 0),
                    (0, 255),
                    (0, 255),
                    (255, 255),
                    (255, 0),
                      
                    (255, 0),
                    (0, 0),
                    (0, 255),
                    (0, 255),
                    (255, 255),
                    (255, 0),
                      
                    (255, 0),
                    (0, 0),
                    (0, 255),
                    (0, 255),
                    (255, 255),
                    (255, 0),
                      
                    (0, 0),
                    (255, 0),
                    (255, 255),
                    (255, 255),
                    (0, 255),
                    (0, 0)]

        mf = cl.mem_flags
        
        for coord in tex_coords:
            self.texc.append([coord[0], coord[1],0,0])
        np_texc = np.array(self.texc, dtype=np.float32)
        #print(self.texc)
        self.tex_coords = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_texc)
        
        for vert in self.np_points:
            vertices.append([vert[0], vert[1], vert[2]])
        points = []
        for v in vertices:
            points.append((v[0], v[1], v[2], 1.0))
        self.np_points = np.array(points, dtype=np.float32)
        #print(self.np_points.shape)
        
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
        for colour in c:
            colours.append(colour)
        for colour in self.np_colours:
            colours.append(colour)
        #print(colours)
        self.np_colours = np.array(colours, dtype="uint")
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
    
#     def make_tiles(self, tile_size):
#         np_tris = np.empty_like(self.cl_out)
#         cl.enqueue_copy(queue, np_tris, self.cl_out)
#         p = multiprocessing.Pool()
#         tiles = []
#         result = p.map(tri_tiles, tiles)
#         
#     def tri_tiles(n, tris, tile_size)
        
    def render(self, render_surface, font):
        np_model = np.eye(4, dtype=np.float32)
        Math3D.translate(np_model, (1.0, 1.0, 1.0))
        np_model = Math3D.rotate(np_model, 10.0 * self.rotation[0], (1.0, 0.0, 0.0))
        np_model = Math3D.rotate(np_model, 10.0 * self.rotation[1], (0.0, 1.0, 0.0))
        
        view = np.eye(4, dtype=np.float32)
        Math3D.translate(view, (-self.viewpos[0], -self.viewpos[1], self.viewpos[2]))
        
        right = 12.0
        top = 8.0
        far = 100.0
        near = 0.1
        
        orth_proj = np.eye(4, dtype=np.float32)
        orth_proj[0][0] = 1 / right
        orth_proj[1][1] = 1 / top
        orth_proj[2][2] = -2 / (far - near)
        orth_proj[2][3] = -((far + near) / (far - near))
        
        np_view = np.dot(orth_proj, view)
        np_view = np.dot(np_view, np_model)
        
        mf = cl.mem_flags
        self.cl_points = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.np_points)
        self.cl_view = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_view)
        self.cl_model = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_model)
        self.cl_out = cl.Buffer(self.ctx, mf.READ_WRITE, self.np_points.nbytes)
        
        self.vertex_shader(self.queue, (self.np_points.shape[0],), None, self.cl_points, self.cl_view, self.cl_screen, self.cl_out)

        y = round(self.h / 100)
        x = round(self.w / 100)
        tilesizex = cl.cltypes.uint(self.w/x)
        tilesizey = cl.cltypes.uint(self.h/y)
        self.cl_tile_maps = cl.Buffer(self.ctx, mf.READ_WRITE, (4*y*x*self.np_points.shape[0]))
        self.cl_tile_layer = cl.Buffer(self.ctx, mf.READ_WRITE, (4*y*x))
        self.cl_tile_layers = cl.Buffer(self.ctx, mf.READ_WRITE, (4*y*x*self.np_points.shape[0]))
        self.make_tiles1(self.queue, (self.np_points.shape[0], x, y), None, self.cl_out, self.cl_tile_maps, self.cl_tile_layers, tilesizex, tilesizey)
        self.make_tiles2(self.queue, (x,y), None, self.cl_tile_maps, self.cl_tile_layers, self.cl_tile_layer, cl.cltypes.uint(self.np_points.shape[0]))
        
#         np_out = np.empty((self.np_points.shape[0], x, y), dtype=np.int32)
#         cl.enqueue_copy(self.queue, np_out, self.cl_tile_layers)
#         print(np_out)
        
#         np_out = np.empty((x, y), dtype=np.int32)
#         cl.enqueue_copy(self.queue, np_out, self.cl_tile_layer)
#         print(np_out)

        self.fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        self.dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=(self.h, self.w))
        self.prg.draw_tris(self.queue, (self.h, self.w), None, self.cl_out, self.tex_coords, cl.cltypes.uint(self.np_points.shape[0]), self.cl_colours, self.cl_tile_layers, self.cl_tile_layer, self.tex, self.dest_buf, tilesizex, tilesizey).wait()
        self.dest = np.empty((self.w,self.h,4), dtype="uint8")
        cl.enqueue_copy(self.queue, self.dest, self.dest_buf, origin=(0, 0), region=(self.h, self.w))

        surf = pygame.surfarray.make_surface(self.dest[:,:,:3])
        render_surface.blit(surf, (0, 0))
        verts = font.render(str(len(self.np_points)), 1, (0, 0, 0))
        render_surface.blit(verts, (0, 30))
#         print(4*y*x*self.np_points.shape[0])
#         cl.enqueue_fill_buffer(self.queue, self.cl_tile_layers, np.int32(-1), 0, self.cl_tile_layers.size)