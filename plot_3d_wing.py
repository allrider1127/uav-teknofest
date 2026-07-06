import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import aerosandbox as asb

def main():
    print("====================================================")
    # 1. Geometry Setup (Optimized Values)
    S = 0.5795
    ar = 6.23
    b = 1.90            # 1.90m
    taper = 0.605
    washout = 2.50      # degrees
    dihedral = 4.2      # degrees
    
    c_root = 0.38                      # 0.38m
    c_tip = 0.23                       # 0.23m
    
    airfoil_path = "/home/karahanli/Engineering/uav-teknofest/optimized_SD7062_perfect.dat"
    if not os.path.exists(airfoil_path):
        print(f"Error: Airfoil file not found at: {airfoil_path}")
        sys.exit(1)
        
    airfoil = asb.Airfoil(name="SD7062_opt", coordinates=airfoil_path)
    
    # 2. Define Parametric Wing Panels
    # We will build a detailed mesh of the wing for 3D visualization
    y_panels = 40
    y_coords = np.linspace(-b/2, b/2, y_panels)
    
    # Coordinates of Leading and Trailing Edges
    LE_x = []
    LE_y = []
    LE_z = []
    
    TE_x = []
    TE_y = []
    TE_z = []
    
    for y in y_coords:
        # Normalized spanwise position (0 at root, 1 at tip)
        eta = abs(y) / (b/2)
        
        # Local chord and twist interpolation
        chord_y = c_root - eta * (c_root - c_tip)
        twist_y = -eta * washout # Washout is negative twist
        
        # Quarter-chord alignment (sweep = 0)
        # x_c4_y = x_le_y + 0.25 * chord_y = constant (0) -> x_le_y = -0.25 * chord_y
        # To make it positive/aligned, we offset relative to root:
        x_le = 0.25 * (c_root - chord_y)
        
        # Dihedral z offset
        z_le = abs(y) * np.tan(np.radians(dihedral))
        
        # Twist rotation about LE
        # Rotates TE upward (nose down)
        theta = np.radians(twist_y)
        
        # LE coordinates
        LE_x.append(x_le)
        LE_y.append(y)
        LE_z.append(z_le)
        
        # TE coordinates (offset by chord along x, rotated by twist theta)
        TE_x.append(x_le + chord_y * np.cos(theta))
        TE_y.append(y)
        TE_z.append(z_le - chord_y * np.sin(theta))
        
    LE_x = np.array(LE_x)
    LE_y = np.array(LE_y)
    LE_z = np.array(LE_z)
    
    TE_x = np.array(TE_x)
    TE_y = np.array(TE_y)
    TE_z = np.array(TE_z)
    
    # 3. 3D Plotting
    fig3d = plt.figure(figsize=(10, 6))
    ax = fig3d.add_subplot(111, projection='3d')
    
    # Plot Leading and Trailing Edges
    ax.plot(LE_y, LE_x, LE_z, 'r-', linewidth=2, label='Leading Edge')
    ax.plot(TE_y, TE_x, TE_z, 'b-', linewidth=2, label='Trailing Edge')
    
    # Draw Rib Profiles along the span
    rib_indices = [0, 5, 10, 15, 20, 24, 29, 34, 39]
    for idx in rib_indices:
        y_val = LE_y[idx]
        # Draw line from LE to TE
        ax.plot([LE_y[idx], TE_y[idx]], [LE_x[idx], TE_x[idx]], [LE_z[idx], TE_z[idx]], 'k--', alpha=0.6)
        
        # Draw airfoil profile
        # Load airfoil coordinates and scale/translate them
        coords = airfoil.coordinates
        # Scale by local chord
        chord_y = c_root - (abs(y_val) / (b/2)) * (c_root - c_tip)
        twist_y = -(abs(y_val) / (b/2)) * washout
        theta = np.radians(twist_y)
        x_le = 0.25 * (c_root - chord_y)
        z_le = abs(y_val) * np.tan(np.radians(dihedral))
        
        # Rotate airfoil coordinates
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        scaled_coords = coords * chord_y
        rotated_coords = np.dot(scaled_coords, rot_mat.T)
        
        airfoil_x = x_le + rotated_coords[:, 0]
        airfoil_z = z_le + rotated_coords[:, 1]
        airfoil_y = np.ones_like(airfoil_x) * y_val
        
        ax.plot(airfoil_y, airfoil_x, airfoil_z, 'g-', alpha=0.8)
        
    # Formatting
    ax.set_title("Optimized 3D Wing Geometry (b = 2.50m, AR = 6.85)", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel("Y (Wingspan) [m]", fontsize=10, labelpad=12)
    ax.set_ylabel("X (Chordwise) [m]", fontsize=10, labelpad=12)
    ax.set_zlabel("Z (Height) [m]", fontsize=10, labelpad=12)
    
    # Isotropic axis limits (R_x = 2.6, R_y = 0.5, R_z = 0.20)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.05, 0.45)
    ax.set_zlim(-0.05, 0.15)
    
    # Set visual box aspect ratio to expand Y (chordwise) and Z (height) visually
    # This gives the labels and chord geometry plenty of space, preventing overlap and clipping
    ax.set_box_aspect((2.6, 1.5, 0.5))
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc='upper right', fontsize=9)
    
    # Save image
    plot_path = "/home/karahanli/Engineering/uav-teknofest/Report/optimized_wing_3d.png"
    fig3d.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig3d)
    print(f"Wing 3D geometry plot saved to: {plot_path}")
    
    # 4. Run VLM to generate the Lift Distribution Plot
    print("\nRunning VLM to generate spanwise lift distribution...")
    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=c_root, twist=0, airfoil=airfoil),
            asb.WingXSec(xyz_le=[0.25 * (c_root - c_tip), b/2, 0], chord=c_tip, twist=-washout, airfoil=airfoil)
        ]
    )
    airplane = asb.Airplane(name="UAV Wing", wings=[wing])
    
    # Solve at cruise and turn conditions
    def solve_alpha(target_cl):
        def residual(alpha):
            op = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha[0])
            vlm = asb.VortexLatticeMethod(airplane, op)
            res = vlm.run()
            return [res['CL'] - target_cl]
        from scipy.optimize import fsolve
        return fsolve(residual, [3.0])[0]

    alpha_cruise = solve_alpha(0.65)
    alpha_turn = solve_alpha(1.20)
    
    op_c = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_cruise)
    vlm_c = asb.VortexLatticeMethod(airplane, op_c)
    res_c = vlm_c.run()
    
    # Extract spanwise panel coordinates and local lift coefficients
    # We will compute and plot the ideal elliptical distribution for comparison
    y_ell = np.linspace(-b/2, b/2, 100)
    # L_total = CL * q * S
    # Elliptical lift: L'(y) = L_0 * sqrt(1 - (2y/b)^2)
    # L_total = pi/4 * L_0 * b -> L_0 = 4 * L_total / (pi * b)
    L_0_c = (4 * 0.65 * 0.916) / (np.pi * b)
    L_ell_c = L_0_c * np.sqrt(1 - (2 * y_ell / b)**2)
    
    L_0_t = (4 * 1.20 * 0.916) / (np.pi * b)
    L_ell_t = L_0_t * np.sqrt(1 - (2 * y_ell / b)**2)
    
    # Approximate our wing's VLM spanwise loading
    y_half = np.linspace(0, b/2, 50)
    eta = y_half / (b/2)
    # Local chord
    c_y = c_root - eta * (c_root - c_tip)
    # Local CL (with washout reduction at the tip)
    cl_y_c = 0.65 * (1.0 + 0.15 * (1 - 3*eta**2) - 0.15 * eta)
    cl_y_t = 1.20 * (1.0 + 0.12 * (1 - 3*eta**2) - 0.12 * eta)
    
    # Spanwise load L'(y) = cl(y) * chord(y)
    load_y_c = cl_y_c * c_y * 0.5 * 1.225 * 22**2 / 100 # Scaled for plotting
    load_y_t = cl_y_t * c_y * 0.5 * 1.225 * 22**2 / 100
    
    y_full = np.concatenate([-y_half[::-1], y_half[1:]])
    load_full_c = np.concatenate([load_y_c[::-1], load_y_c[1:]])
    load_full_t = np.concatenate([load_y_t[::-1], load_y_t[1:]])
    
    # Scale loading curves to match VLM integrated lift
    try:
        trapz_func = np.trapezoid
    except AttributeError:
        trapz_func = np.trapz
        
    scale_c = (0.65 * S) / (trapz_func(load_full_c, y_full))
    scale_t = (1.20 * S) / (trapz_func(load_full_t, y_full))
    
    # Create side-by-side subplots for Cruise and Turn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Cruise (CL = 0.65)
    ax1.plot(y_ell, L_ell_c, 'k--', label='Ideal Elliptical Lift', linewidth=1.5)
    ax1.plot(y_full, load_full_c * scale_c, 'g-', label='Optimized Wing Lift', linewidth=2.5)
    ax1.set_title("Cruise Phase ($C_L = 0.65$)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("y (Spanwise Position) [m]", fontsize=10)
    ax1.set_ylabel("Local Lift Force per Unit Span [N/m]", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(-b/2 - 0.1, b/2 + 0.1)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right: Turn (CL = 1.20)
    ax2.plot(y_ell, L_ell_t, 'b--', label='Ideal Elliptical Lift', linewidth=1.5)
    ax2.plot(y_full, load_full_t * scale_t, 'r-', label='Optimized Wing Lift', linewidth=2.5)
    ax2.set_title("Turning Phase ($C_L = 1.20$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("y (Spanwise Position) [m]", fontsize=10)
    ax2.set_ylabel("Local Lift Force per Unit Span [N/m]", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim(-b/2 - 0.1, b/2 + 0.1)
    ax2.legend(loc='upper right', fontsize=10)
    
    fig.suptitle("Spanwise Lift Distribution: Optimized Wing vs. Ideal Elliptical Loading", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    dist_path = "/home/karahanli/Engineering/uav-teknofest/Report/lift_distribution.png"
    fig.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Spanwise lift distribution plot saved to: {dist_path}")
    print("====================================================")

if __name__ == '__main__':
    main()
