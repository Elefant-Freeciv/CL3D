//CL//

uint4 get_bool_map_addr(uint3 addr)
{
    uint4 rtn;
    rtn.x = addr.x;
    rtn.y = addr.y;
    rtn.z = addr.z / 32;
    rtn.w = addr.z % 32;
    return rtn;
}

bool get_tile_map_val(uint3 addr, __global tile_map *bool_map)
{
    uint4 a = get_bool_map_addr(addr);
    uint v = bool_map[a.z][a.x][a.y];
    uint mask = 1 << a.w;
    return (v & mask) >> a.w;
    
}

/*void set_tile_map_val(uint3 addr, __global tile_map *bool_map, bool val)
{
    uint4 a = get_bool_map_addr(addr);
    uint v = bool_map[a.z][a.x][a.y];
    uint mask = 1 << a.w;
    uint rtn = v | mask;
    atomic_xchg(&bool_map[a.z][a.x][a.y], rtn);// = rtn;
}*/

void set_tile_map_val(uint3 addr, __global tile_map *bool_map, bool val)
{
    uint4 a = get_bool_map_addr(addr);
    //uint v = bool_map[a.z][a.x][a.y];
    uint mask = val << a.w;
    //uint rtn = v | mask;
    atomic_or(&bool_map[a.z][a.x][a.y], mask);
}

        
float4 mul(__constant float4 mat[4], const float4 point)
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

bool point_in_triangle (ushort2 pos, float4 v1, float4 v2, float4 v3)
{
    float2 pt = convert_float2(pos);
    return ((orient(v1, v2, pt) && orient(v2, v3, pt) && orient(v3, v1, pt))||(!orient(v1, v2, pt) && !orient(v2, v3, pt) && !orient(v3, v1, pt)));
}

bool lines_intersect(float4 p1, float4 p2, int4 tilerect)
{
    float2 p3 = convert_float2(tilerect.xy);
    float2 p4 = convert_float2(tilerect.zw);
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

float pixel_depth(ushort2 pos, float4 p1, float4 p2, float4 p3)
{
    float2 px = (float2)(convert_float(pos.x),convert_float(pos.y));
    float3 bary = barycentric(px, p1, p2, p3);
    float z = bary.x * 1/p1.w + bary.y * 1/p2.w + bary.z * 1/p3.w;
    return 1/z;
}

uchar4 texture_pixel(ushort2 pos, int i, float z, __global tex_img tex, __constant float8 *tex_coords, float4 p1, float4 p2, float4 p3)
{   
    float2 px = convert_float2(pos);
    float3 bary = barycentric(px, p1, p2, p3);
    float2 st0 = tex_coords[i].s01;
    float2 st1 = tex_coords[i].s23;
    float2 st2 = tex_coords[i].s45;
    
    st0[0] /= p1.w, st0[1] /= p1.w;
    st1[0] /= p2.w, st1[1] /= p2.w;
    st2[0] /= p3.w, st2[1] /= p3.w;
    
    float x = bary.x * st0[0] + bary.y * st1[0] + bary.z * st2[0];
    float y = bary.x * st0[1] + bary.y * st1[1] + bary.z * st2[1];
    x *= z, y *= z;
    
    if(x>=0 && x<=1023 && y>=0 && y<=1023)
    {
        return tex[convert_int(tex_coords[i][6])][min(convert_int(x), 1023)][min(convert_int(y), 1023)];
    }
    else
    {
        return (uchar4)(255, 0, 180, 0);
    }
}

__kernel void vertex(__constant float4 *points,
                     __constant float4 mat[4],
                     __constant float2 *screen,
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

void bool_map_copy(__global bool_layer out, bool_layer in)
{
    for (int i = 0; i <= tilecount.x; i++)
    {
        for (int j = 0; j <= tilecount.y; j++)
        {
            if (in[i][j])
            {
                out[i][j] = 1;
            }
        }
    }
}

float axis_intersect(bool x_or_y,
            float4 p1,
            float4 p2,
            int offset
            )
{
    float m = (p1.y-p2.y)/(p1.x-p2.x);
    float b = -m*p1.x+p1.y;
    if (x_or_y)
    {
        return (m*offset)+b;
    }
    else
    {
        return (offset-b)/m;
    }
}

__kernel void count_tiles(__global bool_layer *bool_map, __global tile_layer *tri_count, uint tcount)
{
    uint2 tile = (uint2)(get_global_id(0), get_global_id(1));
    uint slice = get_global_id(2);
    int j = 0;
    int i_max = min((slice+1)*slice_size, tcount);
    for (int i = slice*slice_size; i<i_max; i++)
    {
        if (bool_map[i][tile.x][tile.y]==1)
        {
            j++;
        }
    }
    tri_count[slice][tile.x][tile.y]=j;
}

__kernel void cumulative_sum(__global tile_layer *tri_count, uint slices)
{
    uint2 tile = (uint2)(get_global_id(0), get_global_id(1));
    int j = 0;
    int temp;
    for(int q = 0; q < slices; q++)
    {
        temp = tri_count[q][tile.x][tile.y];
        tri_count[q][tile.x][tile.y] = j;
        j += temp;
    }
}

__kernel void make_tiles2(__global tile_layer *bool_map, __global tile_layer *out, __global tile_layer *tri_count, __global tile_layer *tri_count_summed, uint tcount)
{
    uint2 tile = (uint2)(get_global_id(0), get_global_id(1));
    uint slice = get_global_id(2);
    int j = tri_count_summed[slice][tile.x][tile.y];
    int i;
    uint j_max;
    uint i_max;
    i = slice*slice_size;
    j_max = tri_count[slice][tile.x][tile.y]+j;
    i_max = min((slice+1)*slice_size, tcount);
    while (j<j_max && i<i_max)
    {
        if (get_tile_map_val((uint3)(tile.x, tile.y, i), bool_map)==1)
        {
            out[j][tile.x][tile.y]=i;
            j++;
        }
        i++;
    }
}

__kernel void make_tiles_stage_4_bb(__global const uint4 *tris,
                                 __global const float4 *points,
                                 __global tile_map *bool_map,
                                 __global tile_layer *tri_count)
{
    int gid = get_global_id(0);
    uint4 tri = tris[gid];
    float4 p1 = points[tri.x];
    float4 p2 = points[tri.y];
    float4 p3 = points[tri.z];
    int4 tbb = (int4)(min(min(p1.x,p2.x), p3.x)/(tilesize.x),
                          min(min(p1.y,p2.y), p3.y)/(tilesize.y), 
                          max(max(p1.x,p2.x), p3.x)/(tilesize.x), 
                          max(max(p1.y,p2.y), p3.y)/(tilesize.y));
    
    int ign;
    for (int i = tbb.x; i<=tbb.z; i++)
    {
        for (int j = tbb.y; j<=tbb.w; j++)
        {
            //bool_map[gid][i][j] = 1;
            set_tile_map_val((uint3)(i, j, gid), bool_map, 1);
            ign = atomic_inc(&tri_count[gid/slice_size][i][j]);
        }
    }
}

__kernel void draw_tris(
    __constant uint4 *tris,
    __constant float4 *points,
    __constant float8 *tex_coords,
    __global const uchar4 *colours,
    __global tile_layer *tile_maps,
    __global tile_layer tri_counts,
    __global tex_img tex,
    __global scr_img screen)
    {
        ushort2 pos = (ushort2)(get_global_id(0), get_global_id(1));
        ushort2 tile = (ushort2)(get_group_id(0),get_group_id(1));
        int tri_count = tri_counts[tile.x][tile.y];
        screen[pos.x][pos.y] = (uchar4)(255,255,255,255);//(tile.x*2.5,tile.y*2.5,255,255));//(uint4)(pos.x,pos.y,convert_int(tris[0].x),255));
        float old_pixel_depth = 100000;
        float test_pixel_depth;
        uchar4 colour;
        for (int i = 0; i<tri_count; i++)
        {
            int tris_index = tile_maps[i][tile.x][tile.y];
            float4 p1 = points[tris[tris_index].x];
            float4 p2 = points[tris[tris_index].y];
            float4 p3 = points[tris[tris_index].z];
            if(point_in_triangle(pos, p1, p2, p3))
            {
                test_pixel_depth = pixel_depth(pos, p1, p2, p3);
                if(test_pixel_depth < old_pixel_depth)
                {
                    if (tex_coords[tris_index].s7 == 1)
                    {
                        colour = colours[tris_index];
                    }
                    else
                    {
                        colour = texture_pixel(pos, tris_index, test_pixel_depth, tex, tex_coords, p1, p2, p3);
                    }
                    // custom fragment shader here
                    //colour /= (convert_uint(test_pixel_depth*10));
                    if (colour[3] != 0)
                    {
                        screen[pos.x][pos.y] = colour;
                    }
                    old_pixel_depth = test_pixel_depth;
                }
            }
        }
    }
    
__kernel void array_sum(__global tile_layer *tri_count, __global tile_layer *tri_count_summed, __global tile_layer out, int loop)
    {
        int2 tile = (int2)(get_global_id(0), get_global_id(1));
        int val = 0;
        for(int i = 0; i < loop; i++)
        {
            val = val+tri_count[i][tile.x][tile.y];
            tri_count_summed[i][tile.x][tile.y]=tri_count[i][tile.x][tile.y];
        }
        out[tile.x][tile.y]=val;
    }
