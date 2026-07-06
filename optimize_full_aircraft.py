import os
import sys
import numpy as np
import aerosandbox as asb
from scipy.optimize import fsolve

# ==========================================
# 1. Geometry and Airfoil Setup
# ==========================================
airfoil_path = "/home/karahanli/Engineering/uav-teknofest/optimized_SD7062_perfect.dat"
if not os.path.exists(airfoil_path):
    print("Error: optimized_SD7062_perfect.dat not found!")
    sys.exit(1)

main_airfoil = asb.Airfoil(name="SD7062_opt", coordinates=airfoil_path)
tail_airfoil = asb.Airfoil(name="naca0012") # Symmetric NACA 0012 for tail

# New Main Wing Parameters (Locked)
b_wing = 1.90 # m
S_wing = 0.5795 # m^2
c_root = 0.38 # m
c_tip = 0.23 # m
taper = c_tip / c_root # 0.605
dihedral_wing = 4.2 # degrees
washout_wing = 2.50 # degrees

# ==========================================
# 2. Solver Helpers
# ==========================================
def solve_alpha_for_cl(airplane, target_cl, initial_guess=3.0):
    def residual(alpha):
        op = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha[0])
        vlm = asb.VortexLatticeMethod(airplane, op, spanwise_resolution=8, chordwise_resolution=8)
        res = vlm.run()
        return [res['CL'] - target_cl]
    
    try:
        alpha_opt = fsolve(residual, [initial_guess])[0]
    except Exception:
        alpha_opt = initial_guess
    return alpha_opt

# ==========================================
# 3. Winglet Performance Sweep
# ==========================================
def evaluate_wing_with_winglet(w_height, w_cant, w_taper=0.5):
    # Winglet geometry
    # Cant angle is measured from horizontal (90 deg = vertical, 75 deg = slightly outward)
    cant_rad = np.radians(w_cant)
    
    # Winglet root is at the wingtip
    y_w_root = b_wing / 2
    z_w_root = y_w_root * np.tan(np.radians(dihedral_wing))
    x_w_root = 0.25 * (c_root - c_tip) # quarter-chord sweep = 0
    
    # Winglet tip
    y_w_tip = y_w_root + w_height * np.cos(cant_rad)
    z_w_tip = z_w_root + w_height * np.sin(cant_rad)
    
    # Tip chord of winglet is tapered
    c_w_tip = c_tip * w_taper
    
    # Winglet sweep (we sweep the winglet trailing edge or quarter-chord)
    # Let's align winglet quarter-chord with wing tip quarter-chord (sweep = 0)
    x_w_tip = x_w_root + 0.25 * (c_tip - c_w_tip)
    
    # Main Wing
    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=c_root, twist=0, airfoil=main_airfoil),
            asb.WingXSec(xyz_le=[x_w_root, y_w_root, z_w_root], chord=c_tip, twist=-washout_wing, airfoil=main_airfoil)
        ]
    )
    
    # Winglet modeled as a separate Wing to avoid node-sharing singularities
    winglet = asb.Wing(
        name="Winglets",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[x_w_root, y_w_root, z_w_root], chord=c_tip, twist=-washout_wing, airfoil=main_airfoil),
            asb.WingXSec(xyz_le=[x_w_tip, y_w_tip, z_w_tip], chord=c_w_tip, twist=-washout_wing, airfoil=main_airfoil)
        ]
    )
    
    airplane = asb.Airplane(name="UAV Winglet", wings=[wing, winglet])
    
    # Evaluate at Cruise Cl = 0.428
    alpha_c = solve_alpha_for_cl(airplane, 0.428, initial_guess=2.0)
    op_c = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_c)
    vlm_c = asb.VortexLatticeMethod(airplane, op_c, spanwise_resolution=8, chordwise_resolution=8)
    res_c = vlm_c.run()
    cdi_cruise = res_c['CD']
    
    # Evaluate at Turning Cl = 0.857
    alpha_t = solve_alpha_for_cl(airplane, 0.857, initial_guess=5.0)
    op_t = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_t)
    vlm_t = asb.VortexLatticeMethod(airplane, op_t, spanwise_resolution=8, chordwise_resolution=8)
    res_t = vlm_t.run()
    cdi_turn = res_t['CD']
    
    return cdi_cruise, cdi_turn

# ==========================================
# 4. Clean Wing Evaluator
# ==========================================
def evaluate_clean_wing():
    wing = asb.Wing(
        name="Clean Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=c_root, twist=0, airfoil=main_airfoil),
            asb.WingXSec(xyz_le=[0.25 * (c_root - c_tip), b_wing / 2, b_wing / 2 * np.tan(np.radians(dihedral_wing))], chord=c_tip, twist=-washout_wing, airfoil=main_airfoil)
        ]
    )
    airplane = asb.Airplane(name="UAV Clean Wing", wings=[wing])
    
    alpha_c = solve_alpha_for_cl(airplane, 0.428, initial_guess=2.0)
    op_c = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_c)
    vlm_c = asb.VortexLatticeMethod(airplane, op_c, spanwise_resolution=8, chordwise_resolution=8)
    res_c = vlm_c.run()
    cdi_cruise = res_c['CD']
    
    alpha_t = solve_alpha_for_cl(airplane, 0.857, initial_guess=5.0)
    op_t = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_t)
    vlm_t = asb.VortexLatticeMethod(airplane, op_t, spanwise_resolution=8, chordwise_resolution=8)
    res_t = vlm_t.run()
    cdi_turn = res_t['CD']
    
    return cdi_cruise, cdi_turn

# ==========================================
# 5. V-Tail Sizing & Stability Optimization
# ==========================================
def evaluate_stability(tail_area, tail_dihedral, tail_boom=0.95):
    # Tail boom is measured from wing root leading edge to tail root leading edge
    x_tail_root = 0.095 + tail_boom
    
    # Tail parameters
    taper_tail = 0.60
    # Aspect ratio of tail as 4.0
    AR_tail = 4.0
    b_tail = np.sqrt(AR_tail * tail_area)
    c_t_root = 2 * tail_area / (b_tail * (1 + taper_tail))
    c_t_tip = c_t_root * taper_tail
    
    # Projected coordinates based on V-tail dihedral angle
    dihedral_rad = np.radians(tail_dihedral)
    y_t_tip = b_tail/2 * np.cos(dihedral_rad)
    z_t_tip = b_tail/2 * np.sin(dihedral_rad)
    
    x_t_tip = x_tail_root + 0.25 * (c_t_root - c_t_tip)
    
    # Build complete airplane with Wing + V-tail
    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=c_root, twist=0, airfoil=main_airfoil),
            asb.WingXSec(xyz_le=[0.25*(c_root-c_tip), b_wing/2, b_wing/2*np.tan(np.radians(dihedral_wing))], chord=c_tip, twist=-washout_wing, airfoil=main_airfoil)
        ]
    )
    
    vtail = asb.Wing(
        name="V-Tail",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[x_tail_root, 0, 0], chord=c_t_root, twist=0, airfoil=tail_airfoil),
            asb.WingXSec(xyz_le=[x_t_tip, y_t_tip, z_t_tip], chord=c_t_tip, twist=0, airfoil=tail_airfoil)
        ]
    )
    
    airplane = asb.Airplane(name="UAV Complete", wings=[wing, vtail])
    
    # Run stability analysis at cruise Cl = 0.428
    alpha_c = solve_alpha_for_cl(airplane, 0.428, initial_guess=2.0)
    op = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_c)
    vlm = asb.VortexLatticeMethod(airplane, op, spanwise_resolution=8, chordwise_resolution=8)
    res = vlm.run()
    
    # Numerical derivatives for stability
    # 1. Pitch stability: dCM/dalpha (Cma)
    op_p = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_c + 1.0)
    vlm_p = asb.VortexLatticeMethod(airplane, op_p, spanwise_resolution=8, chordwise_resolution=8)
    res_p = vlm_p.run()
    Cma = (res_p['Cm'] - res['Cm']) # per degree
    
    # 2. Yaw stability: dCN/dbeta (Cnb)
    op_y = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_c, beta=1.0)
    vlm_y = asb.VortexLatticeMethod(airplane, op_y, spanwise_resolution=8, chordwise_resolution=8)
    res_y = vlm_y.run()
    Cnb = res_y['Cn'] # Cn at beta = 1 deg is dCn/dbeta per degree
    
    # 3. Roll stability: dCl/dbeta (Clb)
    Clb = res_y['Cl'] # Cl at beta = 1 deg is dCl/dbeta per degree
    
    return Cma, Cnb, Clb, res['CD'], c_t_root, c_t_tip, b_tail

# ==========================================
# 5. Optimization Executions
# ==========================================
def main():
    print("\n--- PHASE 1: WINGLET GEOMETRY OPTIMIZATION ---")
    best_cant = 0
    best_height = 0
    min_obj = 999.0
    
    print("Running parametric sweep for winglet cant angle and height...")
    # Sweep cant angle from 45 to 90 degrees, and height from 0.04m to 0.12m
    for cant in np.linspace(45, 90, 10):
        for h in np.linspace(0.04, 0.12, 5):
            cdi_c, cdi_t = evaluate_wing_with_winglet(h, cant)
            obj = 0.60 * cdi_c + 0.40 * cdi_t
            if obj < min_obj:
                min_obj = obj
                best_cant = cant
                best_height = h
                
    # Evaluated drag for wing without winglet
    cdi_c_base, cdi_t_base = evaluate_clean_wing()
    base_obj = 0.60 * cdi_c_base + 0.40 * cdi_t_base
    
    cdi_c_opt, cdi_t_opt = evaluate_wing_with_winglet(best_height, best_cant)
    opt_obj = 0.60 * cdi_c_opt + 0.40 * cdi_t_opt
    
    print("\nWinglet Sweep Results:")
    print(f"Optimal Cant Angle: {best_cant:.1f} degrees (outward tilt)")
    print(f"Optimal Winglet Height: {best_height*1000:.1f} mm")
    print(f"Cruise Induced Drag (with winglet): {cdi_c_opt:.6f} (Baseline: {cdi_c_base:.6f}, {((cdi_c_base-cdi_c_opt)/cdi_c_base)*100:.2f}% reduction)")
    print(f"Turning Induced Drag (with winglet): {cdi_t_opt:.6f} (Baseline: {cdi_t_base:.6f}, {((cdi_t_base-cdi_t_opt)/cdi_t_base)*100:.2f}% reduction)")
    print(f"Overall Multi-Point CDi Reduction: {((base_obj - opt_obj)/base_obj)*100:.2f}%")
    
    print("\n--- PHASE 2: V-TAIL AREA AND DIHEDRAL OPTIMIZATION ---")
    print("Varying V-tail area and dihedral angle to satisfy pitch/yaw stability...")
    best_area = 0.0
    best_d = 0.0
    min_tail_area = 999.0
    
    # Stability thresholds in units per degree (converted from target margins in radians)
    target_Cma = -0.05 / 57.3 # -0.00087 per deg
    target_Cnb = 0.04 / 57.3  # 0.00070 per deg
    
    for area in np.linspace(0.05, 0.15, 20): # m^2
        for dihedral in np.linspace(30, 55, 20): # degrees
            Cma, Cnb, Clb, cd, ctr, ctt, bt = evaluate_stability(area, dihedral)
            if Cma <= target_Cma and Cnb >= target_Cnb:
                if area < min_tail_area:
                    min_tail_area = area
                    best_area = area
                    best_d = dihedral
                    
    Cma_opt, Cnb_opt, Clb_opt, cd_opt, ctr_opt, ctt_opt, bt_opt = evaluate_stability(best_area, best_d)
    
    print("\nV-Tail Sizing Results:")
    print(f"Optimal V-Tail Dihedral Angle: {best_d:.1f} degrees")
    print(f"Optimal V-Tail Projected Area (S_tail): {best_area:.4f} m^2 ({(best_area/S_wing)*100:.1f}% of wing area)")
    print(f"V-Tail Span (b_tail): {bt_opt:.3f} m")
    print(f"V-Tail Root Chord (c_t_root): {ctr_opt:.3f} m")
    print(f"V-Tail Tip Chord (c_t_tip): {ctt_opt:.3f} m")
    print(f"Static Stability Margins:")
    print(f"  Pitch Stability (Cma): {Cma_opt*57.3:.4f} per rad (Target: <= -0.05)")
    print(f"  Yaw Stability (Cnb): {Cnb_opt*57.3:.4f} per rad (Target: >= 0.04)")
    print(f"  Roll Stability (Clb): {Clb_opt*57.3:.4f} per rad (Negative values show positive dihedral stability)")
    
    # Save optimized values to a text file for reporting
    with open("optimized_aircraft_values.txt", "w") as f:
        f.write(f"winglet_height={best_height}\n")
        f.write(f"winglet_cant={best_cant}\n")
        f.write(f"winglet_cruise_cdi={cdi_c_opt}\n")
        f.write(f"winglet_turn_cdi={cdi_t_opt}\n")
        f.write(f"vtail_area={best_area}\n")
        f.write(f"vtail_dihedral={best_d}\n")
        f.write(f"vtail_span={bt_opt}\n")
        f.write(f"vtail_root_chord={ctr_opt}\n")
        f.write(f"vtail_tip_chord={ctt_opt}\n")
        f.write(f"Cma={Cma_opt*57.3}\n")
        f.write(f"Cnb={Cnb_opt*57.3}\n")
        f.write(f"Clb={Clb_opt*57.3}\n")
        
    print("\nOptimized values successfully saved to 'optimized_aircraft_values.txt'.")

if __name__ == '__main__':
    main()
