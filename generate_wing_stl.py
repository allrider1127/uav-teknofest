import os
import sys
import numpy as np

def load_airfoil(filepath):
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.array(coords)

def main():
    print("Generating 3D STL mesh for optimized wing...")
    airfoil_path = "/home/karahanli/Engineering/uav-teknofest/optimized_SD7062_perfect.dat"
    if not os.path.exists(airfoil_path):
        print(f"Error: Airfoil file not found at: {airfoil_path}")
        sys.exit(1)
        
    # Coords in Selig format (TE -> upper LE -> TE lower)
    coords = load_airfoil(airfoil_path)
    N_af = len(coords)
    
    # Parameters
    S = 0.916
    ar = 6.85
    b = np.sqrt(ar * S) # 2.50m
    taper = 0.768
    washout = 2.49      # degrees
    dihedral = 4.2      # degrees
    
    c_root = 2 * S / (b * (1 + taper)) # 0.414m
    c_tip = c_root * taper             # 0.318m
    
    # Spanwise grid: M sections from left tip (-b/2) to right tip (b/2)
    M_span = 80
    y_coords = np.linspace(-b/2, b/2, M_span)
    
    # Generate all vertices: grid of size (M_span, N_af, 3)
    vertices = np.zeros((M_span, N_af, 3))
    
    for i, y in enumerate(y_coords):
        eta = abs(y) / (b/2)
        chord_y = c_root - eta * (c_root - c_tip)
        twist_y = eta * washout # Twist is positive washout (trailing edge up)
        
        # Quarter-chord alignment (sweep = 0)
        x_le = 0.25 * (c_root - chord_y)
        z_le = abs(y) * np.tan(np.radians(dihedral))
        
        # Rotation angle for twist (radians)
        theta = np.radians(twist_y)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for j in range(N_af):
            x_af = coords[j, 0]
            z_af = coords[j, 1] # y coordinate in Selig represents thickness/z
            
            # Scale
            x_scaled = x_af * chord_y
            z_scaled = z_af * chord_y
            
            # Rotate (clockwise for positive washout -> trailing edge moves up)
            x_rot = x_scaled * cos_t - z_scaled * sin_t
            z_rot = x_scaled * sin_t + z_scaled * cos_t
            
            # Translate to geometry position
            vertices[i, j, 0] = x_le + x_rot # X is chordwise (forward to backward)
            vertices[i, j, 1] = y            # Y is spanwise
            vertices[i, j, 2] = z_le + z_rot # Z is vertical
            
    # Write ASCII STL file
    stl_path = "/home/karahanli/Engineering/uav-teknofest/Report/optimized_wing_3d.stl"
    print(f"Writing STL to: {stl_path}...")
    
    with open(stl_path, 'w') as f:
        f.write("solid optimized_uav_wing\n")
        
        # Generate triangular facets
        for i in range(M_span - 1):
            for j in range(N_af - 1):
                # Vertex indices for the quad:
                # v1: (i, j), v2: (i, j+1), v3: (i+1, j), v4: (i+1, j+1)
                p1 = vertices[i, j]
                p2 = vertices[i, j+1]
                p3 = vertices[i+1, j]
                p4 = vertices[i+1, j+1]
                
                # Triangle 1 (p1, p2, p3)
                # Compute normal
                v1 = p2 - p1
                v2 = p3 - p1
                normal1 = np.cross(v1, v2)
                norm1_mag = np.linalg.norm(normal1)
                if norm1_mag > 1e-9:
                    normal1 /= norm1_mag
                else:
                    normal1 = np.array([0.0, 0.0, 1.0])
                    
                f.write(f"facet normal {normal1[0]:.7e} {normal1[1]:.7e} {normal1[2]:.7e}\n")
                f.write("  outer loop\n")
                f.write(f"    vertex {p1[0]:.7f} {p1[1]:.7f} {p1[2]:.7f}\n")
                f.write(f"    vertex {p2[0]:.7f} {p2[1]:.7f} {p2[2]:.7f}\n")
                f.write(f"    vertex {p3[0]:.7f} {p3[1]:.7f} {p3[2]:.7f}\n")
                f.write("  endloop\n")
                f.write("endfacet\n")
                
                # Triangle 2 (p3, p2, p4)
                v3 = p2 - p3
                v4 = p4 - p3
                normal2 = np.cross(v3, v4)
                norm2_mag = np.linalg.norm(normal2)
                if norm2_mag > 1e-9:
                    normal2 /= norm2_mag
                else:
                    normal2 = np.array([0.0, 0.0, 1.0])
                    
                f.write(f"facet normal {normal2[0]:.7e} {normal2[1]:.7e} {normal2[2]:.7e}\n")
                f.write("  outer loop\n")
                f.write(f"    vertex {p3[0]:.7f} {p3[1]:.7f} {p3[2]:.7f}\n")
                f.write(f"    vertex {p2[0]:.7f} {p2[1]:.7f} {p2[2]:.7f}\n")
                f.write(f"    vertex {p4[0]:.7f} {p4[1]:.7f} {p4[2]:.7f}\n")
                f.write("  endloop\n")
                f.write("endfacet\n")
                
        f.write("endsolid optimized_uav_wing\n")
        
    print("STL generation complete! File saved successfully.")

if __name__ == '__main__':
    main()
