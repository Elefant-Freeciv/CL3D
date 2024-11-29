import pyopencl as cl
import pygame, math
import multiprocessing

import numpy as np
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
    def __init__(self, h, w, target_tile_size=25):
        self.h = h
        self.w = w
        self.y = round(self.h / target_tile_size)
        self.x = round(self.w / target_tile_size)
        self.tilesizex = cl.cltypes.uint(self.w/self.x)
        self.tilesizey = cl.cltypes.uint(self.h/self.y)
        if (self.tilesizex*self.x)<self.w:
            self.tilesizex += 1
        if (self.tilesizey*self.y)<self.h:
            self.tilesizey += 1
        self.viewpos = [0.0, 0.0, -10.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.delta = 0.0
        self.clicking = False
        self.start_click = []
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx,
        f'''

        #define tilesize (uint2)({self.tilesizey}, {self.tilesizex})

        typedef int tile_layer[{self.y}][{self.x}];
        typedef uint4 scr_img[{self.h}][{self.w}];
        typedef uint4 tex_img[256][256];
        '''+'''//CL//

        
        
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
            float m1 = (p3.y-p4.y)/(p3.x-p4.x);
            float m2 = (p1.y-p2.y)/(p1.x-p2.x);
            float b1 = -m1*p3.x+p3.y;
            float b2 = -m2*p1.x+p1.y;
            float x = (b2-b1)/(m1-m2);
            float y = m1*x+b1;
            bool a = (x >= p3.x && x <= p4.x);
            bool b = (y >= min(p3.y, p4.y) && y <= max(p4.y, p3.y));
            bool c = (x >= min(p1.x, p2.x) && x <= max(p1.x, p2.x));
            if (a && b && c){return true;}
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
        
        uint4 texture_pixel(int2 pos, int i, float z, __global tex_img tex, __global float8 *tex_coords, float4 p1, float4 p2, float4 p3)
        {   
            float2 px = (float2)(convert_float(pos.x),convert_float(pos.y));
            float3 bary = barycentric(px, p1, p2, p3);
            //printf("%i", i);
            float2 st0 = tex_coords[i].s01;
            float2 st1 = tex_coords[i].s23;
            float2 st2 = tex_coords[i].s45;
            
            st0[0] /= p1.w, st0[1] /= p1.w;
            st1[0] /= p2.w, st1[1] /= p2.w;
            st2[0] /= p3.w, st2[1] /= p3.w;
            
            float x = bary.x * st0[0] + bary.y * st1[0] + bary.z * st2[0];
            float y = bary.x * st0[1] + bary.y * st1[1] + bary.z * st2[1];
            x *= z, y *= z;
            
            return tex[convert_int(x)][convert_int(y)];
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
                                    __global const uint4 *tris,
                                    __global const float4 *points,
                                    __global tile_layer *bool_map,
                                    __global tile_layer *tile_layers
                                 )
        {
            int gid = get_global_id(0);
            uint4 tri = tris[gid];
            float4 p1 = points[tri.x];
            float4 p2 = points[tri.y];
            float4 p3 = points[tri.z];
            int2 tile = (int2)(get_global_id(1), get_global_id(2));
            bool_map[gid][tile.x][tile.y] = 0;
            tile_layers[gid][tile.x][tile.y] = 0;
            bool a, b, c, d, e, f; 
            int4 tilerect = (int4)(tile.x*tilesize.x, tile.y*tilesize.y, tile.x*tilesize.x+tilesize.x, tile.y*tilesize.y+tilesize.y);
            a = (p1.x >= tilerect.x && p1.x <= tilerect.z);
            b = (p1.y >= tilerect.y && p1.y <= tilerect.w);
            c = (p2.x >= tilerect.x && p2.x <= tilerect.z);
            d = (p2.y >= tilerect.y && p2.y <= tilerect.w);
            e = (p3.x >= tilerect.x && p3.x <= tilerect.z);
            f = (p3.y >= tilerect.y && p3.y <= tilerect.w);
            if ((a && b) || (c && d) || (e && f)){bool_map[gid][tile.x][tile.y] = 1;}
            a = point_in_triangle((int2)(tilerect.x, tilerect.y), p1, p2, p3);
            b = point_in_triangle((int2)(tilerect.x, tilerect.w), p1, p2, p3);
            c = point_in_triangle((int2)(tilerect.z, tilerect.y), p1, p2, p3);
            d = point_in_triangle((int2)(tilerect.z, tilerect.w), p1, p2, p3);
            if (a || b || c || d){bool_map[gid][tile.x][tile.y] = 1;}
            a = lines_intersect(p1, p2, tilerect);
            b = lines_intersect(p2, p3, tilerect);
            c = lines_intersect(p3, p1, tilerect);
            d = lines_intersect(p1, p2, (int4)(tilerect.x,tilerect.w,tilerect.z,tilerect.y));
            e = lines_intersect(p2, p3, (int4)(tilerect.x,tilerect.w,tilerect.z,tilerect.y));
            f = lines_intersect(p3, p1, (int4)(tilerect.x,tilerect.w,tilerect.z,tilerect.y));
            if (a || b || c || d || e || f){bool_map[gid][tile.x][tile.y] = 1;}
        }
        
        __kernel void make_tiles2(__global tile_layer *bool_map, __global tile_layer *out, __global tile_layer tri_count, uint tcount)
        {
            int2 tile = (int2)(get_global_id(0), get_global_id(1));
            tri_count[tile.x][tile.y]=0;
            int j = 0;
            for (int i = 0; i<(tcount); i++)
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
            __global const uint4 *tris,
            __global const float4 *points,
            __global const float8 *tex_coords,
            uint pcount,
            __global const uint4 *colours,
            __global tile_layer *tile_maps,
            __global tile_layer tri_count,
            __global tex_img tex,
            __global scr_img screen)
            {
                int2 pos = (int2)(get_global_id(0), get_global_id(1));
                int2 tile = (int2)(pos.x/tilesize.x, pos.y/tilesize.y);
                screen[pos.x][pos.y] = (uint4)(255,255,255,255);//(tile.x*2.5,tile.y*2.5,255,255));//(uint4)(pos.x,pos.y,convert_int(tris[0].x),255));
                float old_pixel_depth = 100000;
                float test_pixel_depth;
                for (int i = 0; i<(tri_count[tile.x][tile.y]); i++)
                {
                    float4 p1 = points[tris[tile_maps[i][tile.x][tile.y]].x];
                    float4 p2 = points[tris[tile_maps[i][tile.x][tile.y]].y];
                    float4 p3 = points[tris[tile_maps[i][tile.x][tile.y]].z];
                    if(point_in_triangle(pos, p1, p2, p3))
                    {
                        test_pixel_depth = pixel_depth(pos, p1, p2, p3);
                        if(test_pixel_depth < old_pixel_depth)
                        {
                            uint4 colour = texture_pixel(pos, tile_maps[i][tile.x][tile.y], test_pixel_depth, tex, tex_coords, p1, p2, p3);
                            // custom fragment shader here
                            //colour /= (convert_uint(test_pixel_depth*10));
                            screen[pos.x][pos.y] = colour;
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
        triangles = [(0,1,2,0),#x axis
                (3,4,5,0),#y axis
                (6,7,8,0)]#z axis
        tex_coords = [(0, 0, 255, 0, 0, 255,1,0),
                    (0, 0, 255, 0, 0, 255,1,0),
                    (0, 0, 255, 0, 0, 255,1,0)]
        colours = [(255,0,0,255),
                   (0,255,0,255),
                   (0,0,255,255)]
        self.np_colours = np.array(colours, dtype="uint")
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
        #print(self.np_points)
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
                 [0.0, 1.0, 0.0, 1.0],
                 [0.0, 0.0, 1.0, 1.0],
                 [0.0, 0.0, 0.0, 1.0]]
        np_model = np.array(model, dtype=np.float32)
#         print(np_model)
        self.src_img = Image.open('Tex2.png').convert('RGBA')
        self.src = np.array(self.src_img, dtype=cl.cltypes.uint)
        print(self.src.shape)
        self.tex = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.src)
        
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
#         vertices=[[-1.0, -1.0, -1.0],
#                  [1.0, -1.0, -1.0],
#                  [1.0, 1.0, -1.0],
#                  [1.0, 1.0, -1.0],
#                  [-1.0, 1.0, -1.0],
#                  [-1.0, -1.0, -1.0],
# 
#                  [-1.0, -1.0, 1.0],
#                  [1.0, -1.0, 1.0],
#                  [1.0, 1.0, 1.0],
#                  [1.0, 1.0, 1.0],
#                  [-1.0, 1.0, 1.0],
#                  [-1.0, -1.0, 1.0],
# 
#                  [-1.0, 1.0, 1.0],
#                  [-1.0, 1.0, -1.0],
#                  [-1.0, -1.0, -1.0],
#                  [-1.0, -1.0, -1.0],
#                  [-1.0, -1.0, 1.0],
#                  [-1.0, 1.0, 1.0],
# 
#                  [1.0, 1.0, 1.0],
#                  [1.0, 1.0, -1.0],
#                  [1.0, -1.0, -1.0],
#                  [1.0, -1.0, -1.0],
#                  [1.0, -1.0, 1.0],
#                  [1.0, 1.0, 1.0],
# 
#                  [-1.0, -1.0, -1.0],
#                  [1.0, -1.0, -1.0],
#                  [1.0, -1.0, 1.0],
#                  [1.0, -1.0, 1.0],
#                  [-1.0, -1.0, 1.0],
#                  [-1.0, -1.0, -1.0],
# 
#                  [-1.0, 1.0, -1.0],
#                  [1.0, 1.0, -1.0],
#                  [1.0, 1.0, 1.0],
#                  [1.0, 1.0, 1.0],
#                  [-1.0, 1.0, 1.0],
#                  [-1.0, 1.0, -1.0]]
#         tex_coords = [(255, 0),
#                     (0, 0),
#                     (0, 255),
#                     (0, 255),
#                     (0, 0),
#                     (255, 0),
#                       
#                     (255, 0),
#                     (0, 0),
#                     (0, 255),
#                     (0, 255),
#                     (255, 255),
#                     (255, 0),
#                       
#                     (255, 0),
#                     (0, 0),
#                     (0, 255),
#                     (0, 255),
#                     (255, 255),
#                     (255, 0),
#                       
#                     (255, 0),
#                     (0, 0),
#                     (0, 255),
#                     (0, 255),
#                     (255, 255),
#                     (255, 0),
#                       
#                     (255, 0),
#                     (0, 0),
#                     (0, 255),
#                     (0, 255),
#                     (255, 255),
#                     (255, 0),
#                       
#                     (0, 0),
#                     (255, 0),
#                     (255, 255),
#                     (255, 255),
#                     (0, 255),
#                     (0, 0)]

        vertices = [(-1,-1,-1),#0
                    (1,-1,-1),#1
                    (-1,1,-1),#2
                    (1,1,-1),#3
                    (-1,-1,1),#4
                    (-1,1,1),#5
                    (1,-1,1),#6
                    (1,1,1)]#7
        triangles = [(0,1,2),
                    (2,1,3)]
#                     (4,5,6),
#                     (5,6,7),
#                     (0,1,4),
#                     (1,4,6),
#                     (2,3,5),
#                     (3,5,7),
#                     (1,3,7),
#                     (1,6,7),
#                     (0,2,4),
#                     (4,2,5)]
        
        tex_coords = [(0, 0, 255, 0, 0, 255,1,0),
                      (0, 255, 255, 0, 255,255,1,0)]
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0),
#                       (0, 0, 255, 0, 0, 255,1,0)
#                       ]
        mf = cl.mem_flags
        
        tris = []
        tcount = self.np_points.shape[0]
        for tri in self.np_tris:
            tris.append((tri[0], tri[1], tri[2], 1))
        for tri in triangles:
            tris.append((tri[0]+tcount, tri[1]+tcount, tri[2]+tcount, 1))
        self.np_tris = np.array(tris, dtype=cl.cltypes.uint)
        print(self.np_tris)
        self.cl_tris = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_tris)
        
        for coord in self.np_tex_coords:
            self.texc.append(coord)
        for coord in tex_coords:
            self.texc.append(coord)
        self.np_tex_coords = np.array(self.texc, dtype=np.float32)
        self.tex_coords = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_tex_coords)
        
        points = []
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
        for colour in c:
            colours.append(colour)
        for colour in self.np_colours:
            colours.append(colour)
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

        self.mapsize = int(self.np_tris.shape[0])
        self.cl_tile_maps = cl.Buffer(self.ctx, mf.READ_WRITE, (4*self.y*self.x*self.mapsize))
        self.cl_tile_layer = cl.Buffer(self.ctx, mf.READ_WRITE, (4*self.y*self.x))
        self.cl_tile_layers = cl.Buffer(self.ctx, mf.READ_WRITE, (4*self.y*self.x*self.mapsize))
        self.make_tiles1(self.queue, (self.mapsize, self.y, self.x), None, self.cl_tris, self.cl_out, self.cl_tile_maps, self.cl_tile_layers)
        self.make_tiles2(self.queue, (self.y,self.x), None, self.cl_tile_maps, self.cl_tile_layers, self.cl_tile_layer, cl.cltypes.uint(self.np_tris.shape[0]))
        
        np_out = np.empty((self.y, self.x), dtype=np.int32)
        cl.enqueue_copy(self.queue, np_out, self.cl_tile_layer)

        self.dest = np.empty((self.h,self.w,4), dtype=cl.cltypes.uint)
        self.dest_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dest)
        self.prg.draw_tris(self.queue, (self.h, self.w), None, self.cl_tris, self.cl_out, self.tex_coords, cl.cltypes.uint(self.np_points.shape[0]), self.cl_colours, self.cl_tile_layers, self.cl_tile_layer, self.tex, self.dest_buf).wait()
        cl.enqueue_copy(self.queue, self.dest, self.dest_buf)

        surf = pygame.surfarray.make_surface(self.dest[:,:,:3])
        surf = pygame.transform.rotate(surf, 90)
        surf = pygame.transform.flip(surf, False, True)
        render_surface.blit(surf, (0, 0))
        verts = font.render(str(len(self.np_points)), 1, (0, 0, 0))
        render_surface.blit(verts, (0, 30))
        for i in range(self.y):
            for j in range(self.x):
                render_surface.blit(font.render(str(np_out[i][j]), 1, (0, 0, 0)), (j*self.tilesizex, i*self.tilesizey))