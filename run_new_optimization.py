import os
import sys
import urllib.request
import subprocess
import shutil
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
import aerosandbox as asb
from scipy.optimize import fsolve

# ==========================================
# 1. 2D Airfoil Optimization Setup
# ==========================================

# CST Definition
def class_function(x):
    return np.sqrt(x) * (1.0 - x)

def shape_function(x, weights):
    n = len(weights) - 1
    S = np.zeros_like(x)
    for i, w in enumerate(weights):
        basis = comb(n, i) * (x**i) * ((1.0 - x)**(n - i))
        S += w * basis
    return S

def cst_airfoil(x, w_up, w_low, y_te=0.0, delta_y_te=0.001):
    y_up = class_function(x) * shape_function(x, w_up) + x * (y_te + 0.5 * delta_y_te)
    y_low = class_function(x) * shape_function(x, w_low) + x * (y_te - 0.5 * delta_y_te)
    return y_up, y_low

def get_thickness(w_up, w_low):
    x = np.linspace(0, 1, 200)
    y_up, y_low = cst_airfoil(x, w_up, w_low)
    return np.max(y_up - y_low)

# XFOIL Interface for 2D Multi-Point
def evaluate_airfoil_xfoil(w_up, w_low, run_id, Re=460000, cl_cruise=0.428, cl_turn=0.857):
    x_grid = np.linspace(0, 1, 100)
    x_cos = 0.5 * (1.0 - np.cos(np.pi * x_grid))
    y_up, y_low = cst_airfoil(x_cos, w_up, w_low)
    
    x_coords = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_coords = np.concatenate([y_up[::-1], y_low[1:]])

    coord_file = f"temp_opt_{run_id}.dat"
    polar_file = f"temp_polar_{run_id}.txt"
    input_file = f"temp_input_{run_id}.txt"

    with open(coord_file, 'w') as f:
        f.write(f"CST_OPT_{run_id}\n")
        for xi, yi in zip(x_coords, y_coords):
            f.write(f"  {xi:.7f}   {yi:.7f}\n")

    if os.path.exists(polar_file):
        os.remove(polar_file)

    commands = [
        f"load {coord_file}",
        "pane",
        "oper",
        "v",
        str(Re),
        "iter 120",
        f"pacc",
        f"{polar_file}",
        "",
        "aseq -5 16 0.5",
        f"pacc",
        "quit"
    ]

    with open(input_file, 'w') as f:
        f.write("\n".join(commands) + "\n")

    try:
        cmd = ["xvfb-run", "-a", "xfoil"] if shutil.which("xvfb-run") else ["xfoil"]
        subprocess.run(cmd, stdin=open(input_file, 'r'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if os.path.exists(input_file): os.remove(input_file)
        if os.path.exists(coord_file): os.remove(coord_file)

    success = False
    cl_max = -999.0
    cd_cruise = 999.0
    cd_turn = 999.0
    cm_cruise = 999.0

    if os.path.exists(polar_file):
        try:
            with open(polar_file, 'r') as f:
                lines = f.readlines()
            
            data_started = False
            alphas, cls, cds, cms = [], [], [], []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 7 and data_started:
                    try:
                        alphas.append(float(parts[0]))
                        cls.append(float(parts[1]))
                        cds.append(float(parts[2]))
                        cms.append(float(parts[4]))
                    except ValueError:
                        continue
                elif len(parts) > 0 and parts[0] == "alpha" and "CL" in parts:
                    data_started = True
            
            if len(cls) > 0:
                cls = np.array(cls)
                cds = np.array(cds)
                cms = np.array(cms)
                
                cl_max = max(cls)
                success = True
                
                if cl_cruise >= min(cls) and cl_cruise <= max(cls):
                    cd_cruise = np.interp(cl_cruise, cls, cds)
                    cm_cruise = np.interp(cl_cruise, cls, cms)
                else:
                    cd_cruise = 0.05
                    
                if cl_turn >= min(cls) and cl_turn <= max(cls):
                    cd_turn = np.interp(cl_turn, cls, cds)
                else:
                    cd_turn = 0.10
        except Exception:
            pass
        finally:
            if os.path.exists(polar_file): os.remove(polar_file)

    return cl_max, cd_cruise, cd_turn, cm_cruise, success

run_counter = 0

def robust_objective(x_vector):
    global run_counter
    run_counter += 1
    
    mid = len(x_vector) // 2
    w_up = x_vector[:mid]
    w_low = x_vector[mid:]
    
    # 1. Aerodynamic Evaluation (Re=460,000, Cl_c=0.428, Cl_t=0.857)
    cl_max, cd_c, cd_t, cm_c, success = evaluate_airfoil_xfoil(w_up, w_low, run_counter, Re=460000, cl_cruise=0.428, cl_turn=0.857)
    
    # 2. Geometric evaluations
    x_test = np.linspace(0.01, 0.99, 100)
    y_up_test, y_low_test = cst_airfoil(x_test, w_up, w_low)
    thickness = np.max(y_up_test - y_low_test)
    
    penalty = 0.0
    
    # Thickness Constraint: t/c >= 13.5% (to fit the 25mm spar at the 23cm tip chord)
    if thickness < 0.135:
        penalty += (0.135 - thickness) * 200.0
    # Thickness limit: max 14.5% to prevent high form drag
    if thickness > 0.145:
        penalty += (thickness - 0.145) * 100.0
        
    # Prevent self-intersection
    if np.any(y_low_test > y_up_test):
        penalty += 50.0
        
    # Stall limit constraint (Cl_max >= 1.45 to guarantee stall safety margin)
    if cl_max < 1.45:
        penalty += (1.45 - cl_max) * 30.0
        
    # Pitching moment constraint (Cm_cruise >= -0.075)
    if cm_c < -0.075:
        penalty += (-0.075 - cm_c) * 20.0
        
    # --- ROBUSTNESS CONSTRAINTS: Curvature and Inflection Control ---
    # We evaluate numerical derivatives of upper and lower surfaces
    dy_up = np.diff(y_up_test) / np.diff(x_test)
    d2y_up = np.diff(dy_up) / np.diff(x_test[:-1])
    
    # Constraint A: Upper surface second derivative must be strictly negative (concave down)
    # in the middle-aft part of the airfoil (from x = 0.05 to x = 0.85) to prevent reflex/waviness.
    idx_mid_up = (x_test[1:-1] > 0.05) & (x_test[1:-1] < 0.85)
    positive_curvatures = d2y_up[idx_mid_up]
    positive_curvatures = positive_curvatures[positive_curvatures > 0]
    if len(positive_curvatures) > 0:
        penalty += np.sum(positive_curvatures) * 100.0
        
    # Constraint B: Number of sign changes in second derivative (inflection points) must be zero
    # on the upper surface to prevent local waviness.
    signs_up = np.sign(d2y_up[idx_mid_up])
    sign_changes = np.sum(np.abs(np.diff(signs_up)) > 1.5)
    if sign_changes > 0:
        penalty += sign_changes * 10.0
        
    if not success:
        return 1.5  # Heavy penalty for failure to converge
        
    # Multi-point objective: 60% cruise drag + 40% turning drag
    obj_val = 0.60 * cd_c + 0.40 * cd_t + penalty
    return obj_val

def fit_cst(x_coords, y_coords, order=4):
    le_idx = np.argmin(x_coords)
    x_up, y_up = x_coords[:le_idx+1], y_coords[:le_idx+1]
    x_low, y_low = x_coords[le_idx:], y_coords[le_idx:]
    
    up_sort = np.argsort(x_up)
    x_up, y_up = x_up[up_sort], y_up[up_sort]
    
    low_sort = np.argsort(x_low)
    x_low, y_low = x_low[low_sort], y_low[low_sort]
    
    x_fit = np.linspace(0.001, 0.999, 100)
    y_up_interp = np.interp(x_fit, x_up, y_up)
    y_low_interp = np.interp(x_fit, x_low, y_low)
    
    A = np.zeros((len(x_fit), order + 1))
    C = class_function(x_fit)
    for i in range(order + 1):
        A[:, i] = C * comb(order, i) * (x_fit**i) * ((1.0 - x_fit)**(order - i))
        
    w_up, _, _, _ = np.linalg.lstsq(A, y_up_interp, rcond=None)
    w_low, _, _, _ = np.linalg.lstsq(A, y_low_interp, rcond=None)
    return w_up, w_low

# ==========================================
# 2. 3D Wing Twist Optimization Setup
# ==========================================

def solve_alpha_for_cl(airplane, target_cl, initial_guess=3.0):
    def residual(alpha):
        op = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha[0])
        vlm = asb.VortexLatticeMethod(airplane, op)
        res = vlm.run()
        return [res['CL'] - target_cl]
    
    alpha_opt = fsolve(residual, [initial_guess])[0]
    return alpha_opt

def evaluate_wing(taper, washout, airfoil_coords_path):
    # New geometric dimensions
    b = 1.90 # m
    S = 0.5795 # m^2
    c_root = 2 * S / (b * (1 + taper))
    c_tip = c_root * taper
    dihedral = 4.2 # degrees
    
    airfoil = asb.Airfoil(name="SD7062_opt", coordinates=airfoil_coords_path)
    
    # Define wing geometry (quarter-chord sweep = 0)
    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=c_root,
                twist=0,
                airfoil=airfoil
            ),
            asb.WingXSec(
                xyz_le=[0.25 * (c_root - c_tip), b/2, b/2 * np.tan(np.radians(dihedral))],
                chord=c_tip,
                twist=-washout, # Washout is negative twist at tip
                airfoil=airfoil
            )
        ]
    )
    
    airplane = asb.Airplane(name="UAV Wing", wings=[wing])
    
    # 1. Cruise Phase: CL = 0.428
    alpha_cruise = solve_alpha_for_cl(airplane, 0.428, initial_guess=3.0)
    op_c = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_cruise)
    vlm_c = asb.VortexLatticeMethod(airplane, op_c)
    res_c = vlm_c.run()
    cdi_cruise = res_c['CD']
    
    # 2. Turning Phase: CL = 0.857
    alpha_turn = solve_alpha_for_cl(airplane, 0.857, initial_guess=6.0)
    op_t = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_turn)
    vlm_t = asb.VortexLatticeMethod(airplane, op_t)
    res_t = vlm_t.run()
    cdi_turn = res_t['CD']
    
    return cdi_cruise, cdi_turn

# ==========================================
# 3. Main Optimization Loop
# ==========================================

def main():
    print("====================================================")
    print("    TEKNOFEST 7.5kg UAV Aerodynamic Design Loop     ")
    print("====================================================")
    
    # 1. 2D Airfoil optimization
    url = "https://m-selig.ae.illinois.edu/ads/coord/sd7062.dat"
    local_file = "sd7062_tmp.dat"
    try:
        print("\nDownloading baseline coordinates...")
        urllib.request.urlretrieve(url, local_file)
        with open(local_file, 'r') as f:
            lines = f.readlines()
        coords = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 2:
                try: coords.append([float(parts[0]), float(parts[1])])
                except ValueError: continue
        coords = np.array(coords)
    except Exception as e:
        print(f"Error downloading: {e}")
        # Synthetic fallback
        coords = np.array([
            [1.0, 0.0], [0.95, 0.005], [0.9, 0.012], [0.8, 0.028], [0.7, 0.045],
            [0.6, 0.063], [0.5, 0.078], [0.4, 0.091], [0.3, 0.098], [0.2, 0.095],
            [0.1, 0.078], [0.05, 0.058], [0.025, 0.041], [0.0125, 0.029], [0.0, 0.0],
            [0.0125, -0.018], [0.025, -0.024], [0.05, -0.030], [0.1, -0.035], [0.2, -0.039],
            [0.3, -0.041], [0.4, -0.042], [0.5, -0.040], [0.6, -0.036], [0.7, -0.030],
            [0.8, -0.022], [0.9, -0.012], [0.95, -0.006], [1.0, 0.0]
        ])
    
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    if os.path.exists(local_file): os.remove(local_file)
    
    # Fit
    w_up_base, w_low_base = fit_cst(x_coords, y_coords, order=4)
    
    # Baseline 2D performance
    cl_max_b, cd_c_b, cd_t_b, cm_c_b, success_b = evaluate_airfoil_xfoil(
        w_up_base, w_low_base, 0, Re=460000, cl_cruise=0.428, cl_turn=0.857
    )
    print("\n=== Baseline Airfoil Performance ===")
    print(f"Thickness: {get_thickness(w_up_base, w_low_base)*100:.2f}%")
    print(f"Cl_max: {cl_max_b:.4f}, Cruise Cd (0.428): {cd_c_b:.5f}, Turning Cd (0.857): {cd_t_b:.5f}, Cruise Cm: {cm_c_b:.5f}")
    
    # Run SLSQP optimization on 2D coordinates
    x0 = np.concatenate([w_up_base, w_low_base])
    bounds = []
    for w in w_up_base:
        bounds.append((w - 0.25 * abs(w), w + 0.25 * abs(w)))
    for w in w_low_base:
        bounds.append((w - 0.04, w + 0.04))
        
    print("\nStarting SLSQP Airfoil Optimization...")
    res = minimize(
        robust_objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'disp': True, 'maxiter': 30, 'eps': 1e-4}
    )
    
    opt_x = res.x
    mid = len(opt_x) // 2
    w_up_opt = opt_x[:mid]
    w_low_opt = opt_x[mid:]
    
    # Generate coordinates
    x_grid = np.linspace(0, 1, 100)
    x_cos = 0.5 * (1.0 - np.cos(np.pi * x_grid))
    y_up_opt, y_low_opt = cst_airfoil(x_cos, w_up_opt, w_low_opt)
    x_coords_opt = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_coords_opt = np.concatenate([y_up_opt[::-1], y_low_opt[1:]])
    
    airfoil_save_path = "/home/karahanli/Engineering/uav-teknofest/optimized_SD7062_perfect.dat"
    with open(airfoil_save_path, 'w') as f:
        f.write("SD7062_PERFECT_OPTIMIZED_TEKNOFEST_7.5KG\n")
        for xi, yi in zip(x_coords_opt, y_coords_opt):
            f.write(f"  {xi:.7f}   {yi:.7f}\n")
    print(f"\nOptimized airfoil coordinates saved to: {airfoil_save_path}")
    
    cl_max_o, cd_c_o, cd_t_o, cm_c_o, success_o = evaluate_airfoil_xfoil(
        w_up_opt, w_low_opt, 9999, Re=460000, cl_cruise=0.428, cl_turn=0.857
    )
    opt_thick = get_thickness(w_up_opt, w_low_opt)
    
    print("\n=== Optimized Airfoil Performance ===")
    print(f"Thickness: {opt_thick*100:.2f}%")
    print(f"Cl_max: {cl_max_o:.4f}, Cruise Cd (0.428): {cd_c_o:.5f}, Turning Cd (0.857): {cd_t_o:.5f}, Cruise Cm: {cm_c_o:.5f}")
    
    # 2. 3D Planform Washout Optimization
    # Fixed taper ratio (lambda = 0.23 / 0.38 = 0.605)
    taper = 0.23 / 0.38
    print(f"\nOptimizing 3D Wing Washout Twist for wingspan 1.90m, chords 0.38m to 0.23m...")
    
    # Design Variable: Washout angle (in degrees)
    # Objective: Minimize 60% cruise induced drag + 40% turning induced drag
    def wing_objective(washout_var):
        cdi_cruise, cdi_turn = evaluate_wing(taper, washout_var[0], airfoil_save_path)
        return 0.60 * cdi_cruise + 0.40 * cdi_turn
        
    # Bounds: Twist in [1.5, 4.0] degrees
    # Constraint: Twist >= 2.0 degrees to prevent tip stall and ensure root-first stall
    bounds_wing = [(1.5, 4.0)]
    cons_wing = [{'type': 'ineq', 'fun': lambda x: x[0] - 2.0}] # twist - 2.0 >= 0
    
    res_wing = minimize(
        wing_objective,
        [2.5],
        method='SLSQP',
        bounds=bounds_wing,
        constraints=cons_wing,
        options={'disp': True, 'eps': 1e-2}
    )
    
    opt_washout = res_wing.x[0]
    print(f"\nOptimal Washout Twist Angle: {opt_washout:.3f} degrees")
    
    cdi_cruise_opt, cdi_turn_opt = evaluate_wing(taper, opt_washout, airfoil_save_path)
    cdi_cruise_base, cdi_turn_base = evaluate_wing(taper, 2.5, airfoil_save_path) # baseline twist
    
    print("\n====================================================")
    print("               Optimization Summary                 ")
    print("====================================================")
    print(f"Airfoil Thickness:    {opt_thick*100:.2f}%")
    print(f"Airfoil Cl_max:       {cl_max_o:.4f}")
    print(f"Airfoil Cruise Cd:    {cd_c_o:.5f} (Baseline: {cd_c_b:.5f}, {((cd_c_b - cd_c_o)/cd_c_b)*100:.1f}% reduction)")
    print(f"Airfoil Turning Cd:   {cd_t_o:.5f} (Baseline: {cd_t_b:.5f}, {((cd_t_b - cd_t_o)/cd_t_b)*100:.1f}% reduction)")
    print(f"Airfoil Cruise Cm:    {cm_c_o:.5f}")
    print(f"Wing Washout Twist:   {opt_washout:.3f} deg")
    print(f"Wing Cruise CDi:      {cdi_cruise_opt:.6f} (Twist 2.5: {cdi_cruise_base:.6f})")
    print(f"Wing Turning CDi:     {cdi_turn_opt:.6f} (Twist 2.5: {cdi_turn_base:.6f})")
    print("====================================================")
    
    # Save optimized values to a text file for script integration
    with open("optimized_values.txt", "w") as f:
        f.write(f"airfoil_thickness={opt_thick}\n")
        f.write(f"airfoil_cl_max={cl_max_o}\n")
        f.write(f"airfoil_cd_cruise={cd_c_o}\n")
        f.write(f"airfoil_cd_turn={cd_t_o}\n")
        f.write(f"airfoil_cm_cruise={cm_c_o}\n")
        f.write(f"wing_washout_twist={opt_washout}\n")
        f.write(f"wing_cdi_cruise={cdi_cruise_opt}\n")
        f.write(f"wing_cdi_turn={cdi_turn_opt}\n")

if __name__ == '__main__':
    main()
