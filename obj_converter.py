file = open("plane.obj").read().splitlines()
output = open("plane.cl3d", "w")
output.write("[")
for line in file:
    if line.startswith("v "):
        l = line.split()
        output.write("("+l[1]+", "+l[2]+", "+l[3]+"),\n")
#         print(line.split(" "))
output.write("]\n###########################################\n###########################################\n###########################################\n###########################################\n###########################################\n###########################################\n[")

tex_coords = []
for line in file:
    if line.startswith("vt "):
        l = line.split()
        a = float(l[1])
        b = float(l[2])
        a = a*255
        b = b*255
        tex_coords.append((b,a))

tex_coords_sorted = ""
print(tex_coords)
for line in file:
    if line.startswith("f "):
        l = line.split()
        #output.write("("+str(int(l[1].split("/")[0])-1)+", "+str(int(l[2].split("/")[0])-1)+", "+str(int(l[3].split("/")[0])-1)+"),\n")
        #print("("+l[1].split("/")[0]+", "+l[2].split("/")[0]+", "+l[3].split("/")[0]+"),\n")
        #print(l)
        a=tex_coords[int(l[1].split("/")[1])-1]
        b=tex_coords[int(l[2].split("/")[1])-1]
        c=tex_coords[int(l[3].split("/")[1])-1]
        tex_coords_sorted += "("+str(256-a[0])+", "+str(a[1])+", "+str(256-b[0])+", "+str(b[1])+", "+str(256-c[0])+", "+str(c[1])+", 1, 0),\n"

        
print(tex_coords_sorted)
output.write("]\n###########################################\n###########################################\n###########################################\n###########################################\n###########################################\n###########################################\n[")
output.write(tex_coords_sorted)
output.close()
file.close()
