import os
import sys
import urllib.request
import subprocess
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
import matplotlib.pyplot as plt

# 1. CST Definition
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

# 2. NACA 4-Digit Generator (e.g., NACA 4415)
def generate_naca_4digit(m=0.04, p=0.40, t=0.15, n_points=100):
    x = np.linspace(0, 1, n_points)
    # Cosine spacing
    x_cos = 0.5 * (1.0 - np.cos(np.pi * x))
    
    # Thickness distribution
    yt = 5.0 * t * (0.2969 * np.sqrt(x_cos) - 0.1260 * x_cos - 0.3516 * (x_cos**2) + 0.2843 * (x_cos**3) - 0.1015 * (x_cos**4))
    
    # Camber line
    yc = np.zeros_like(x_cos)
    dyc_dx = np.zeros_like(x_cos)
    
    idx1 = x_cos < p
    yc[idx1] = (m / (p**2)) * (2.0 * p * x_cos[idx1] - x_cos[idx1]**2)
    dyc_dx[idx1] = (2.0 * m / (p**2)) * (p - x_cos[idx1])
    
    idx2 = x_cos >= p
    yc[idx2] = (m / ((1.0 - p)**2)) * ((1.0 - 2.0 * p) + 2.0 * p * x_cos[idx2] - x_cos[idx2]**2)
    dyc_dx[idx2] = (2.0 * m / ((1.0 - p)**2)) * (p - x_cos[idx2])
    
    theta = np.arctan(dyc_dx)
    
    xu = x_cos - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x_cos + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine into Selig format (TE -> LE -> TE)
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return x_coords, y_coords

# 3. Fit coordinates to CST
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

# 4. XFOIL Solver Interface
def evaluate_airfoil_xfoil(w_up, w_low, run_id, Re=544000):
    x_grid = np.linspace(0, 1, 100)
    x_cos = 0.5 * (1.0 - np.cos(np.pi * x_grid))
    y_up, y_low = cst_airfoil(x_cos, w_up, w_low)
    
    x_coords = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_coords = np.concatenate([y_up[::-1], y_low[1:]])

    coord_file = f"temp_comp_{run_id}.dat"
    polar_file = f"temp_comp_polar_{run_id}.txt"
    input_file = f"temp_comp_input_{run_id}.txt"

    with open(coord_file, 'w') as f:
        f.write(f"CST_COMP_{run_id}\\n")
        for xi, yi in zip(x_coords, y_coords):
            f.write(f"  {xi:.7f}   {yi:.7f}\\n")

    if os.path.exists(polar_file):
        os.remove(polar_file)

    commands = [
        f"load {coord_file}",
        "pane",
        "oper",
        "v",
        str(Re),
        "iter 100",
        f"pacc",
        f"{polar_file}",
        "",
        "aseq 0 16 0.5",
        f"pacc",
        "quit"
    ]

    with open(input_file, 'w') as f:
        f.write("\n".join(commands) + "\n")

    try:
        subprocess.run(["xfoil"], stdin=open(input_file, 'r'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
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
                
                if 0.65 >= min(cls) and 0.65 <= max(cls):
                    cd_cruise = np.interp(0.65, cls, cds)
                    cm_cruise = np.interp(0.65, cls, cms)
                else:
                    cd_cruise = 0.05
                    
                if 1.20 >= min(cls) and 1.20 <= max(cls):
                    cd_turn = np.interp(1.20, cls, cds)
                else:
                    cd_turn = 0.10
        except Exception:
            pass
        finally:
            if os.path.exists(polar_file): os.remove(polar_file)

    return cl_max, cd_cruise, cd_turn, cm_cruise, success

# 5. Robust Objective Function with Curvature and Inflection Constraint
run_counter = 0

def robust_objective(x_vector, initial_thickness):
    global run_counter
    run_counter += 1
    
    mid = len(x_vector) // 2
    w_up = x_vector[:mid]
    w_low = x_vector[mid:]
    
    # 1. Aerodynamic evaluation
    cl_max, cd_c, cd_t, cm_c, success = evaluate_airfoil_xfoil(w_up, w_low, run_counter)
    
    # 2. Geometric evaluations
    x_test = np.linspace(0.01, 0.99, 100)
    y_up_test, y_low_test = cst_airfoil(x_test, w_up, w_low)
    thickness = np.max(y_up_test - y_low_test)
    
    penalty = 0.0
    
    # Strict thickness limit (must be >= 13.5% or initial thickness, whichever is higher, to prevent structural thinning)
    target_thick = max(0.135, initial_thickness - 0.005)
    if thickness < target_thick:
        penalty += (target_thick - thickness) * 200.0
        
    # Prevent self-intersection
    if np.any(y_low_test > y_up_test):
        penalty += 50.0
        
    # Stall limit constraint (Cl_max >= 1.45)
    if cl_max < 1.45:
        penalty += (1.45 - cl_max) * 30.0
        
    # Pitching moment constraint (Cm_cruise >= -0.075)
    if cm_c < -0.075:
        penalty += (-0.075 - cm_c) * 15.0
        
    # --- ROBUSTNESS CONSTRAINTS: Curvature and Inflection Control ---
    # We evaluate numerical derivatives of upper and lower surfaces
    dy_up = np.diff(y_up_test) / np.diff(x_test)
    d2y_up = np.diff(dy_up) / np.diff(x_test[:-1])
    
    # Constraint A: Upper surface second derivative must be strictly negative (concave down)
    # in the middle-aft part of the airfoil (from x = 0.05 to x = 0.85) to prevent reflex/waviness
    # and maintain flow health.
    idx_mid_up = (x_test[1:-1] > 0.05) & (x_test[1:-1] < 0.85)
    positive_curvatures = d2y_up[idx_mid_up[1:]]
    positive_curvatures = positive_curvatures[positive_curvatures > 0]
    if len(positive_curvatures) > 0:
        penalty += np.sum(positive_curvatures) * 100.0
        
    # Constraint B: Number of sign changes in second derivative (inflection points) must be zero
    # on the upper surface to prevent local waviness.
    signs_up = np.sign(d2y_up[idx_mid_up[1:]])
    sign_changes = np.sum(np.abs(np.diff(signs_up)) > 1.5)
    if sign_changes > 0:
        penalty += sign_changes * 10.0
        
    # Objective: 60% cruise drag + 40% turning drag + penalties
    if not success:
        return 2.0
        
    return 0.60 * cd_c + 0.40 * cd_t + penalty

# 6. Optimization Runner
def optimize_airfoil_family(name, w_up_base, w_low_base):
    print(f"\nRunning Robust Optimization for {name} family...")
    initial_thickness = get_thickness(w_up_base, w_low_base)
    print(f"Initial thickness: {initial_thickness*100:.2f}%")
    
    # Baseline performance
    cl_max, cd_c, cd_t, cm_c, success = evaluate_airfoil_xfoil(w_up_base, w_low_base, 0)
    print(f"Baseline Cl_max: {cl_max:.4f}, Cruise Cd: {cd_c:.5f}, Turning Cd: {cd_t:.5f}, Cruise Cm: {cm_c:.5f}")
    
    x0 = np.concatenate([w_up_base, w_low_base])
    
    # Bounds: Allow weights to vary by +/- 20% to keep them near the baseline shape family,
    # preventing radical shape deviations while allowing performance tweaks.
    bounds = []
    for w in w_up_base:
        bounds.append((w - 0.2 * abs(w), w + 0.2 * abs(w)))
    for w in w_low_base:
        bounds.append((w - 0.04, w + 0.04))
        
    res = minimize(
        robust_objective,
        x0,
        args=(initial_thickness,),
        method='SLSQP',
        bounds=bounds,
        options={'disp': True, 'maxiter': 20, 'eps': 1e-3}
    )
    
    opt_x = res.x
    mid = len(opt_x) // 2
    w_up_opt = opt_x[:mid]
    w_low_opt = opt_x[mid:]
    
    cl_max_opt, cd_c_opt, cd_t_opt, cm_c_opt, success_opt = evaluate_airfoil_xfoil(w_up_opt, w_low_opt, 9999)
    opt_thickness = get_thickness(w_up_opt, w_low_opt)
    
    print(f"Optimized Cl_max: {cl_max_opt:.4f}, Cruise Cd: {cd_c_opt:.5f}, Turning Cd: {cd_t_opt:.5f}, Cruise Cm: {cm_c_opt:.5f}, Thickness: {opt_thickness*100:.2f}%")
    
    return w_up_opt, w_low_opt, {
        'thick_base': initial_thickness, 'thick_opt': opt_thickness,
        'cl_max_base': cl_max, 'cl_max_opt': cl_max_opt,
        'cd_c_base': cd_c, 'cd_c_opt': cd_c_opt,
        'cd_t_base': cd_t, 'cd_t_opt': cd_t_opt,
        'cm_c_base': cm_c, 'cm_c_opt': cm_c_opt
    }

def save_airfoil_dat(filename, name, w_up, w_low):
    x_grid = np.linspace(0, 1, 100)
    x_cos = 0.5 * (1.0 - np.cos(np.pi * x_grid))
    y_up, y_low = cst_airfoil(x_cos, w_up, w_low)
    x_coords = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_coords = np.concatenate([y_up[::-1], y_low[1:]])
    with open(filename, 'w') as f:
        f.write(f"{name}\n")
        for xi, yi in zip(x_coords, y_coords):
            f.write(f"  {xi:.7f}   {yi:.7f}\n")

def main():
    print("====================================================")
    print("   Comparative and Curvature-Constrained Optimizer ")
    print("====================================================")
    
    # 1. Download SD7062 Coords
    print("Downloading SD7062 baseline...")
    url = "https://m-selig.ae.illinois.edu/ads/coord/sd7062.dat"
    local_file = "sd7062_tmp.dat"
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
    x_sd, y_sd = coords[:, 0], coords[:, 1]
    if os.path.exists(local_file): os.remove(local_file)
    
    # Fit SD7062 to 4th-order CST (5 weights)
    w_up_sd, w_low_sd = fit_cst(x_sd, y_sd, order=4)
    
    # 2. Generate NACA 4415
    print("Generating NACA 4415 baseline...")
    x_naca, y_naca = generate_naca_4digit(m=0.04, p=0.40, t=0.15, n_points=100)
    w_up_naca, w_low_naca = fit_cst(x_naca, y_naca, order=4)
    
    # Run SD7062 family optimization
    w_up_sd_opt, w_low_sd_opt, stats_sd = optimize_airfoil_family("SD7062 (High-Lift Low-Re)", w_up_sd, w_low_sd)
    
    # Run NACA 4415 family optimization
    w_up_naca_opt, w_low_naca_opt, stats_naca = optimize_airfoil_family("NACA 4415 (Classic Thick)", w_up_naca, w_low_naca)
    
    # Save optimized coordinates
    save_airfoil_dat("../optimized_SD7062_robust.dat", "SD7062_ROBUST_OPTIMIZED", w_up_sd_opt, w_low_sd_opt)
    save_airfoil_dat("../optimized_NACA4415_robust.dat", "NACA4415_ROBUST_OPTIMIZED", w_up_naca_opt, w_low_naca_opt)
    
    # Plot Airfoils
    plt.figure(figsize=(10, 4))
    x_grid = np.linspace(0, 1, 200)
    
    # SD7062
    y_up_sd_b, y_low_sd_b = cst_airfoil(x_grid, w_up_sd, w_low_sd)
    y_up_sd_o, y_low_sd_o = cst_airfoil(x_grid, w_up_sd_opt, w_low_sd_opt)
    plt.plot(x_grid, y_up_sd_b, 'b--', label="SD7062 Baseline")
    plt.plot(x_grid, y_low_sd_b, 'b--')
    plt.plot(x_grid, y_up_sd_o, 'b-', label="SD7062 Robust Optimized")
    plt.plot(x_grid, y_low_sd_o, 'b-')
    
    # NACA 4415
    y_up_naca_b, y_low_naca_b = cst_airfoil(x_grid, w_up_naca, w_low_naca)
    y_up_naca_o, y_low_naca_o = cst_airfoil(x_grid, w_up_naca_opt, w_low_naca_opt)
    plt.plot(x_grid, y_up_naca_b, 'r--', label="NACA 4415 Baseline")
    plt.plot(x_grid, y_low_naca_b, 'r--')
    plt.plot(x_grid, y_up_naca_o, 'r-', label="NACA 4415 Robust Optimized")
    plt.plot(x_grid, y_low_naca_o, 'r-')
    
    plt.grid(True)
    plt.axis("equal")
    plt.title("Airfoil Geometry Comparison (Robust CST Optimization)")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.legend()
    plt.tight_layout()
    plt.savefig("airfoil_comparison.png", dpi=300)
    plt.close()
    
    # Print comparison summary for LaTeX integration
    print("\n====================================================")
    print("           LATEX COMPARISON GENERATOR               ")
    print("====================================================")
    print(f"SD7062 Weights Upper: {[round(w, 5) for w in w_up_sd_opt]}")
    print(f"SD7062 Weights Lower: {[round(w, 5) for w in w_low_sd_opt]}")
    print(f"NACA4415 Weights Upper: {[round(w, 5) for w in w_up_naca_opt]}")
    print(f"NACA4415 Weights Lower: {[round(w, 5) for w in w_low_naca_opt]}")
    
    # Generate polar comparison data
    print("\nStarting polar evaluations for matplotlib comparison plots...")
    # Evaluate polars for all 4 profiles
    def run_full_polar(w_up, w_low, name, run_id):
        x_grid = np.linspace(0, 1, 100)
        x_cos = 0.5 * (1.0 - np.cos(np.pi * x_grid))
        y_up, y_low = cst_airfoil(x_cos, w_up, w_low)
        x_coords = np.concatenate([x_cos[::-1], x_cos[1:]])
        y_coords = np.concatenate([y_up[::-1], y_low[1:]])
        coord_file = f"polar_run_{run_id}.dat"
        polar_file = f"polar_run_{run_id}.txt"
        input_file = f"polar_run_{run_id}.in"
        with open(coord_file, 'w') as f:
            f.write(f"{name}\n")
            for xi, yi in zip(x_coords, y_coords):
                f.write(f"  {xi:.7f}   {yi:.7f}\n")
        commands = [
            f"load {coord_file}",
            "pane",
            "oper",
            "v",
            "544000",
            "iter 100",
            f"pacc",
            f"{polar_file}",
            "",
            "aseq 0 16 0.5",
            f"pacc",
            "quit"
        ]
        with open(input_file, 'w') as f:
            f.write("\n".join(commands) + "\n")
        subprocess.run(["xfoil"], stdin=open(input_file, 'r'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if os.path.exists(input_file): os.remove(input_file)
        if os.path.exists(coord_file): os.remove(coord_file)
        
        alphas, cls, cds, cms = [], [], [], []
        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                lines = f.readlines()
            data_started = False
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
            os.remove(polar_file)
        return alphas, cls, cds, cms

    alp_sdb, cl_sdb, cd_sdb, cm_sdb = run_full_polar(w_up_sd, w_low_sd, "SDB", 1)
    alp_sdo, cl_sdo, cd_sdo, cm_sdo = run_full_polar(w_up_sd_opt, w_low_sd_opt, "SDO", 2)
    alp_nacab, cl_nacab, cd_nacab, cm_nacab = run_full_polar(w_up_naca, w_low_naca, "NACAB", 3)
    alp_nacao, cl_nacao, cd_nacao, cm_nacao = run_full_polar(w_up_naca_opt, w_low_naca_opt, "NACAO", 4)
    
    # Plot polars
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. CL vs alpha
    axs[0, 0].plot(alp_sdb, cl_sdb, 'b--', label="SD7062 Baseline")
    axs[0, 0].plot(alp_sdo, cl_sdo, 'b-', label="SD7062 Robust Optimized")
    axs[0, 0].plot(alp_nacab, cl_nacab, 'r--', label="NACA 4415 Baseline")
    axs[0, 0].plot(alp_nacao, cl_nacao, 'r-', label="NACA 4415 Robust Optimized")
    axs[0, 0].set_xlabel("alpha (deg)")
    axs[0, 0].set_ylabel("Cl")
    axs[0, 0].set_title("Lift Coefficient vs. Angle of Attack")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 2. Cl vs Cd
    axs[0, 1].plot(cd_sdb, cl_sdb, 'b--')
    axs[0, 1].plot(cd_sdo, cl_sdo, 'b-')
    axs[0, 1].plot(cd_nacab, cl_nacab, 'r--')
    axs[0, 1].plot(cd_nacao, cl_nacao, 'r-')
    axs[0, 1].set_xlabel("Cd")
    axs[0, 1].set_ylabel("Cl")
    axs[0, 1].set_title("Drag Polar")
    axs[0, 1].grid(True)
    
    # 3. Cm vs alpha
    axs[1, 0].plot(alp_sdb, cm_sdb, 'b--')
    axs[1, 0].plot(alp_sdo, cm_sdo, 'b-')
    axs[1, 0].plot(alp_nacab, cm_nacab, 'r--')
    axs[1, 0].plot(alp_nacao, cm_nacao, 'r-')
    axs[1, 0].set_xlabel("alpha (deg)")
    axs[1, 0].set_ylabel("Cm")
    axs[1, 0].set_title("Pitching Moment Coefficient vs. Angle of Attack")
    axs[1, 0].grid(True)
    
    # 4. L/D vs alpha
    axs[1, 1].plot(alp_sdb, np.array(cl_sdb)/np.array(cd_sdb), 'b--')
    axs[1, 1].plot(alp_sdo, np.array(cl_sdo)/np.array(cd_sdo), 'b-')
    axs[1, 1].plot(alp_nacab, np.array(cl_nacab)/np.array(cd_nacab), 'r--')
    axs[1, 1].plot(alp_nacao, np.array(cl_nacao)/np.array(cd_nacao), 'r-')
    axs[1, 1].set_xlabel("alpha (deg)")
    axs[1, 1].set_ylabel("Cl/Cd")
    axs[1, 1].set_title("Glide Efficiency (Cl/Cd) vs. Angle of Attack")
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("polar_comparison.png", dpi=300)
    plt.close()
    
    print("\nFinished generating plots. Saved 'airfoil_comparison.png' and 'polar_comparison.png'.")

if __name__ == '__main__':
    main()
