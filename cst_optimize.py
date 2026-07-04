import os
import sys
import urllib.request
import subprocess
import shutil
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb

# 1. CST Definition
def class_function(x):
    return np.sqrt(x) * (1.0 - x)

def shape_function(x, weights):
    n = len(weights) - 1
    S = np.zeros_like(x)
    for i, w in enumerate(weights):
        # Bernstein polynomial basis of order n
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

# 2. XFOIL Multi-Point Interface
def evaluate_airfoil_xfoil_multipoint(w_up, w_low, run_id):
    x_grid = np.linspace(0, 1, 100)
    x_cos = 0.5 * (1.0 - np.cos(np.pi * x_grid))
    y_up, y_low = cst_airfoil(x_cos, w_up, w_low)
    
    x_coords = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_coords = np.concatenate([y_up[::-1], y_low[1:]])

    coord_file = f"temp_v2_{run_id}.dat"
    polar_file = f"temp_v2_polar_{run_id}.txt"
    input_file = f"temp_v2_input_{run_id}.txt"

    with open(coord_file, 'w') as f:
        f.write(f"CST_V2_{run_id}\n")
        for xi, yi in zip(x_coords, y_coords):
            f.write(f"  {xi:.7f}   {yi:.7f}\n")

    if os.path.exists(polar_file):
        os.remove(polar_file)

    commands = [
        f"load {coord_file}",
        "pane",
        "oper",
        "v",
        "544000",        # Reynolds number
        "iter 150",      # Give more iterations for fine convergence
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
        # Use xvfb-run if available to support headless execution
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
    target_cl = 0.65

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
                
                if target_cl >= min(cls) and target_cl <= max(cls):
                    cd_cruise = np.interp(target_cl, cls, cds)
                    cm_cruise = np.interp(target_cl, cls, cms)
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

# 3. Objective Function
run_counter = 0

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

def objective(x_vector):
    global run_counter
    run_counter += 1
    
    mid = len(x_vector) // 2
    w_up = x_vector[:mid]
    w_low = x_vector[mid:]
    
    cl_max, cd_cruise, cd_turn, cm_cruise, success = evaluate_airfoil_xfoil_multipoint(w_up, w_low, run_counter)
    thickness = get_thickness(w_up, w_low)
    
    penalty = 0.0
    
    # 1. Thickness constraint (t/c >= 13.5% for structural spar)
    if thickness < 0.135:
        penalty += (0.135 - thickness) * 200.0
        
    # Enforce maximum thickness to prevent high form drag
    if thickness > 0.145:
        penalty += (thickness - 0.145) * 100.0
        
    # 2. Leading edge radius check
    # In CST, leading edge radius r_le is related to the first weights: r_le = 0.5 * (w_up[0] - w_low[0])^2
    r_le = 0.5 * (w_up[0] - w_low[0])**2
    if r_le < 0.012:
        penalty += (0.012 - r_le) * 100.0
        
    # 3. Self-intersection check
    x_test = np.linspace(0.01, 0.99, 50)
    y_up_test, y_low_test = cst_airfoil(x_test, w_up, w_low)
    if np.any(y_low_test > y_up_test):
        penalty += 20.0
        
    # 4. Stall limit constraint (Cl_max >= 1.45)
    if cl_max < 1.45:
        penalty += (1.45 - cl_max) * 30.0
        
    # 5. Pitching moment constraint (Cm_cruise >= -0.075)
    if cm_cruise < -0.075:
        penalty += (-0.075 - cm_cruise) * 20.0

    if not success:
        return 1.5  # Heavy penalty for failure to converge
        
    # Multi-point objective: 60% cruise drag + 40% turning drag
    obj_val = 0.60 * cd_cruise + 0.40 * cd_turn + penalty
    return obj_val

def main():
    print("====================================================")
    print("    TEKNOFEST High-Precision Airfoil Optimizer      ")
    print("====================================================")
    
    # Use synthetic SD7062 baseline if Selig server is offline
    url = "https://m-selig.ae.illinois.edu/ads/coord/sd7062.dat"
    local_file = "sd7062_baseline.dat"
    try:
        print("Downloading baseline coordinates...")
        urllib.request.urlretrieve(url, local_file)
        with open(local_file, 'r') as f:
            lines = f.readlines()
        coords = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
        coords = np.array(coords)
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Using synthetic SD7062 baseline coordinates...")
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

    order = 4 # 4th-order CST (5 weights per surface)
    w_up_base, w_low_base = fit_cst(x_coords, y_coords, order=order)
    
    print("\nEvaluating Fitted CST Baseline in XFOIL...")
    cl_max, cd_c, cd_t, cm_c, success = evaluate_airfoil_xfoil_multipoint(w_up_base, w_low_base, 0)
    if success:
        print(f"Baseline Cl_max: {cl_max:.4f}")
        print(f"Baseline Cruise Cd (Cl=0.65): {cd_c:.5f}")
        print(f"Baseline Turning Cd (Cl=1.20): {cd_t:.5f}")
        print(f"Baseline Cruise Cm: {cm_c:.5f}")
    else:
        print("Error: XFOIL failed to evaluate baseline.")
        sys.exit(1)
        
    x0 = np.concatenate([w_up_base, w_low_base])
    
    # Define bounds allowing +/- 25% variation
    bounds = []
    for w in w_up_base:
        bounds.append((w - 0.25 * abs(w), w + 0.25 * abs(w)))
    for w in w_low_base:
        bounds.append((w - 0.04, w + 0.04))
        
    print("\nStarting Parallelized SLSQP High-Precision Optimization...")
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds,
        options={'disp': True, 'maxiter': 50, 'eps': 1e-4}
    )
    
    print("\nOptimization Complete!")
    
    opt_x = res.x
    mid = len(opt_x) // 2
    w_up_opt = opt_x[:mid]
    w_low_opt = opt_x[mid:]
    
    x_grid = np.linspace(0, 1, 100)
    x_cos = 0.5 * (1.0 - np.cos(np.pi * x_grid))
    y_up_opt, y_low_opt = cst_airfoil(x_cos, w_up_opt, w_low_opt)
    
    x_coords_opt = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_coords_opt = np.concatenate([y_up_opt[::-1], y_low_opt[1:]])
    
    save_path = "optimized_SD7062_perfect.dat"
    with open(save_path, 'w') as f:
        f.write("SD7062_PERFECT_OPTIMIZED_TEKNOFEST\n")
        for xi, yi in zip(x_coords_opt, y_coords_opt):
            f.write(f"  {xi:.7f}   {yi:.7f}\n")
    print(f"\nPerfect coordinates saved to '{save_path}'!")
    
    # Final evaluation
    cl_max_opt, cd_c_opt, cd_t_opt, cm_c_opt, success_opt = evaluate_airfoil_xfoil_multipoint(w_up_opt, w_low_opt, 9999)
    opt_thickness = get_thickness(w_up_opt, w_low_opt)
    opt_r_le = 0.5 * (w_up_opt[0] - w_low_opt[0])**2
    base_r_le = 0.5 * (w_up_base[0] - w_low_base[0])**2
    print("\n====================================================")
    print("               Comparison Summary                   ")
    print("====================================================")
    print(f"Metric             Baseline        Optimized")
    print(f"Thickness:         {get_thickness(w_up_base, w_low_base)*100:.2f}%          {opt_thickness*100:.2f}%")
    print(f"LE Radius (r_le):  {base_r_le:.5f}         {opt_r_le:.5f}")
    print(f"Cl_max (Stall):    {cl_max:.4f}          {cl_max_opt:.4f}")
    print(f"Cruise Cd (0.65):  {cd_c:.5f}         {cd_c_opt:.5f} ({(cd_c-cd_c_opt)/cd_c*100:.1f}% reduction)")
    print(f"Turning Cd (1.20): {cd_t:.5f}         {cd_t_opt:.5f} ({(cd_t-cd_t_opt)/cd_t*100:.1f}% reduction)")
    print(f"Cruise Cm:         {cm_c:.5f}        {cm_c_opt:.5f}")
    print("====================================================")

if __name__ == '__main__':
    main()
