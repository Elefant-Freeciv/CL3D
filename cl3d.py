import pyopencl as cl
import pygame, math

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
        
        float my_distance(float4 p1, float4 p2)
        {
            return sqrt(pow((p2.x-p1.x), 2)+pow((p2.y-p1.y), 2));
        }
        
        float pixel_depth(int2 pos, float4 p1, float4 p2, float4 p3)
        {
            float A = p1.y * (p2.w-p3.w) + p2.y * (p3.w-p1.w) + p3.y * (p1.w-p2.w);
            float B = p1.w * (p2.x-p3.x) + p2.w * (p3.x-p1.x) + p3.w * (p1.x-p2.x);
            float C = p1.x * (p2.y-p3.y) + p2.x * (p3.y-p1.y) + p3.x * (p1.y-p2.y);
            float D = -p1.x * (p2.y*p3.w - p3.y*p2.w) - p2.x * (p3.y*p1.w - p1.y*p3.w) - p3.x * (p1.y*p2.w - p2.y*p1.w);
            return ((A * convert_float(pos.x))+(B * convert_float(pos.y))+D)/-C;
        }
        
        uint4 texture_pixel(int2 pos, int i, read_only image2d_t tex, __global float3 *mats)
        {
            float3 mat4[3];
            float2 px = (float2)(convert_float(pos.x),convert_float(pos.y));
                                
            mat4[0] = (float3)(px.x, 0, 1);
            mat4[1] = (float3)(px.y, 0, 1);
            mat4[2] = (float3)(1, 1, 1);
            
            px = (float2)(dot(mats[i], (float3)(mat4[0].x, mat4[1].x, mat4[2].x)), dot(mats[i+1], (float3)(mat4[0].x, mat4[1].x, mat4[2].x)));
            const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
            return read_imageui(tex, sampler, px);
        }

        __kernel void transform(
            __global const float4 *points, __global const float4 mat[4], __global float4 *out
        )
        {
            int gid = get_global_id(0);
            float gidf = convert_float(gid);
            out[gid] = mul(mat, points[gid]);
        }

        __kernel void transform2(
            __global const float4 *points, __global const float4 mat[4], __global float4 *out
        )
        {
            int gid = get_global_id(0);
            float gidf = convert_float(gid);
            out[gid] = mul(mat, points[gid]);
            out[gid].w = -out[gid].z;
            out[gid] = (float4)(out[gid].x / out[gid].w, out[gid].y / out[gid].w, out[gid].z / out[gid].w, points[gid].z);
        }
        
        __kernel void map2screen(
            __global const float4 *points, __global const float2 *screen, __global float4 *out
        )
        {
            int gid = get_global_id(0);
            float x = ((points[gid].x + 1) / 2) * screen[0].x;
            float y = ((-points[gid].y + 1) / 2) * screen[0].y;
            out[gid] = (float4)(y, x, 0, points[gid].w);
        }
        
        __kernel void make_mats(
            __global const float4 *tris, __global const float4 *tex_coords, global float3 *mats
        )
        {
            int i = get_global_id(0)*3;
            float4 p1 = tris[i];
            float4 p2 = tris[i+1];
            float4 p3 = tris[i+2];
            float4 P1 = tex_coords[i];
            float4 P2 = tex_coords[i+1];
            float4 P3 = tex_coords[i+2];
            float3 mat1[3];
            float3 mat2[3];
            float3 row;
            float det;
            /*
            p1.x p2.x p3.x
            p1.y p2.y p3.y
            1    1    1
            */
            row = (float3)((p2.y-p3.y), (p1.y-p3.y), (p1.y-p2.y));
            det = 1/(p1.x*(p2.y-p3.y)-p2.x*(p1.y-p3.y)+p3.x*(p1.y-p2.y));
            
            mat1[0] = (float3)((p2.y-p3.y)*det, -(p2.x-p3.x)*det, (p2.x*p3.y-p2.y*p3.x)*det);
            mat1[1] = (float3)(-(p1.y-p3.y)*det, (p1.x-p3.x)*det, -(p1.x*p3.y-p3.x*p1.y)*det);
            mat1[2] = (float3)((p1.y-p2.y)*det, -(p1.x-p2.x)*det, (p1.x*p2.y-p2.x*p1.y)*det);
            
            mat2[0] = (float3)(P1.x, P2.x, P3.x);
            mat2[1] = (float3)(P1.y, P2.y, P3.y);
            
            mats[i] = (float3)(dot(mat2[0], (float3)(mat1[0].x, mat1[1].x, mat1[2].x)),
                                dot(mat2[0], (float3)(mat1[0].y, mat1[1].y, mat1[2].y)),
                                dot(mat2[0], (float3)(mat1[0].z, mat1[1].z, mat1[2].z)));
            mats[i+1] = (float3)(dot(mat2[1], (float3)(mat1[0].x, mat1[1].x, mat1[2].x)),
                                dot(mat2[1], (float3)(mat1[0].y, mat1[1].y, mat1[2].y)),
                                dot(mat2[1], (float3)(mat1[0].z, mat1[1].z, mat1[2].z)));
        }
        
        __kernel void draw_tris(
            __global const float4 *tris, __global const float4 *tex_coords, uint pcount, __global const uint4 *colours, __global float3 *mats, read_only image2d_t tex, write_only image2d_t screen)
            {
                const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
                int2 pos = (int2)(get_global_id(0), get_global_id(1));
                write_imageui(screen, pos, (uint4)(255,255,255,255));//(uint4)(pos.x,pos.y,convert_int(tris[0].x),255));
                float old_pixel_depth = 100000;
                float test_pixel_depth;
                for (int i = 0; i<pcount; i += 3)
                {
                    
                    if(point_in_triangle(pos, tris[i], tris[i+1], tris[i+2]))
                    {
                        test_pixel_depth = pixel_depth(pos, tris[i], tris[i+1], tris[i+2]);
                        if(test_pixel_depth < old_pixel_depth)
                        {
                            write_imageui(screen, pos, texture_pixel(pos, i, tex, mats));
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
        self.np_mats = np.array((len(vertices), 3), dtype=np.float32)
        self.mats = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.np_mats)
        
        self.knl = self.prg.transform
        self.knl2 = self.prg.transform2
        self.knl3 = self.prg.map2screen
        
        
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
        
        mf = cl.mem_flags
        self.cl_points = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.np_points)
        self.cl_view = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_view)
        self.cl_model = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_model)
        self.cl_out = cl.Buffer(self.ctx, mf.READ_WRITE, self.np_points.nbytes)
        
        self.mats = cl.Buffer(self.ctx, mf.READ_WRITE, self.np_points.nbytes)

        self.knl(self.queue, (self.np_points.shape[0],1), None, self.cl_points, self.cl_model, self.cl_out)
        self.knl2(self.queue, (self.np_points.shape[0],), None, self.cl_out, self.cl_view, self.cl_points)
        self.knl3(self.queue, (self.np_points.shape[0],), None, self.cl_points, self.cl_screen, self.cl_out)
        self.prg.make_mats(self.queue, (int(self.np_points.shape[0]/3),), None, self.cl_out, self.tex_coords, self.mats)#.wait()

        self.fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        self.dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=(self.h, self.w))
        self.prg.draw_tris(self.queue, (self.h, self.w), None, self.cl_out, self.tex_coords, cl.cltypes.uint(self.np_points.shape[0]), self.cl_colours, self.mats, self.tex, self.dest_buf).wait()
        self.dest = np.empty((self.w,self.h,4), dtype="uint8")
        cl.enqueue_copy(self.queue, self.dest, self.dest_buf, origin=(0, 0), region=(self.h, self.w))

        surf = pygame.surfarray.make_surface(self.dest[:,:,:3])
        render_surface.blit(surf, (0, 0))
        verts = font.render(str(len(self.np_points)), 1, (0, 0, 0))
        render_surface.blit(verts, (0, 30))