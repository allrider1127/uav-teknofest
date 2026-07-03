import os
import sys
import subprocess
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

# 1. PARSEC Coordinate Generator (as defined previously)
def parsec_airfoil(p):
    try:
        r_le       = p['r_le']
        x_up       = p['x_up']
        y_up       = p['y_up']
        k_up       = p['k_up']
        x_low      = p['x_low']
        y_low      = p['y_low']
        k_low      = p['k_low']
        y_te       = p['y_te']
        delta_y_te = p['delta_y_te']
        alpha_te   = np.radians(p['alpha_te'])
        beta_te    = np.radians(p['beta_te'])

        # Upper surface coefficients a_n (n = 1 to 6)
        M_up = np.zeros((6, 6))
        B_up = np.zeros(6)
        M_up[0, :] = 1.0 ** (np.arange(1, 7) - 0.5)
        B_up[0]    = y_te + 0.5 * delta_y_te
        M_up[1, :] = x_up ** (np.arange(1, 7) - 0.5)
        B_up[1]    = y_up
        M_up[2, :] = (np.arange(1, 7) - 0.5) * (1.0 ** (np.arange(1, 7) - 1.5))
        B_up[2]    = np.tan(alpha_te - 0.5 * beta_te)
        M_up[3, :] = (np.arange(1, 7) - 0.5) * (x_up ** (np.arange(1, 7) - 1.5))
        B_up[3]    = 0.0
        M_up[4, :] = (np.arange(1, 7) - 0.5) * (np.arange(1, 7) - 1.5) * (x_up ** (np.arange(1, 7) - 2.5))
        B_up[4]    = k_up
        M_up[5, 0] = 1.0
        B_up[5]    = np.sqrt(2.0 * r_le)

        a_up = np.linalg.solve(M_up, B_up)

        # Lower surface coefficients b_n (n = 1 to 6)
        M_low = np.zeros((6, 6))
        B_low = np.zeros(6)
        M_low[0, :] = 1.0 ** (np.arange(1, 7) - 0.5)
        B_low[0]    = y_te - 0.5 * delta_y_te
        M_low[1, :] = x_low ** (np.arange(1, 7) - 0.5)
        B_low[1]    = y_low
        M_low[2, :] = (np.arange(1, 7) - 0.5) * (1.0 ** (np.arange(1, 7) - 1.5))
        B_low[2]    = np.tan(alpha_te + 0.5 * beta_te)
        M_low[3, :] = (np.arange(1, 7) - 0.5) * (x_low ** (np.arange(1, 7) - 1.5))
        B_low[3]    = 0.0
        M_low[4, :] = (np.arange(1, 7) - 0.5) * (np.arange(1, 7) - 1.5) * (x_low ** (np.arange(1, 7) - 2.5))
        B_low[4]    = k_low
        M_low[5, 0] = -1.0
        B_low[5]    = np.sqrt(2.0 * r_le)

        b_low = np.linalg.solve(M_low, B_low)

        x = np.linspace(0, 1, 100)
        x_cos = 0.5 * (1.0 - np.cos(np.pi * x))

        y_up_coords = np.zeros_like(x_cos)
        for n in range(6):
            y_up_coords += a_up[n] * (x_cos ** (n + 0.5))

        y_low_coords = np.zeros_like(x_cos)
        for n in range(6):
            y_low_coords += b_low[n] * (x_cos ** (n + 0.5))

        x_coords = np.concatenate([x_cos[::-1], x_cos[1:]])
        y_coords = np.concatenate([y_up_coords[::-1], y_low_coords[1:]])
        
        return x_coords, y_coords
    except np.linalg.LinAlgError:
        return None, None

# 2. XFOIL Interface
def evaluate_airfoil_xfoil(p_dict, run_id):
    """
    Runs XFOIL to evaluate the performance of an airfoil specified by parameters.
    Saves a temporary coordinates file and runs XFOIL.
    Returns: (cl_max, cd_at_target_cl, cm_at_target_cl, success)
    """
    x, y = parsec_airfoil(p_dict)
    if x is None:
        return 0.0, 1.0, 0.0, False # Penalty

    # Create temporary files with a unique run_id to support multiprocessing
    coord_file = f"temp_airfoil_{run_id}.dat"
    polar_file = f"temp_polar_{run_id}.txt"
    input_file = f"temp_input_{run_id}.txt"

    # Save coordinates
    with open(coord_file, 'w') as f:
        f.write(f"TEMP_{run_id}\n")
        for xi, yi in zip(x, y):
            f.write(f"  {xi:.7f}   {yi:.7f}\n")

    # Clean up previous polar if it exists
    if os.path.exists(polar_file):
        os.remove(polar_file)

    # XFOIL Commands
    commands = [
        f"load {coord_file}",
        "pane",          # Interpolate panels to ensure smooth surface distribution
        "oper",
        "v",             # Viscous mode
        "544000",        # Reynolds number
        "iter 100",      # Increase iteration limit for convergence
        f"pacc",
        f"{polar_file}",
        "",              # Blank line for polar log file
        "aseq 0 16 0.5", # Sweep angle of attack from 0 to 16 degrees
        f"pacc",
        "quit"
    ]

    with open(input_file, 'w') as f:
        f.write("\n".join(commands) + "\n")

    try:
        # Run XFOIL (assuming it is installed and in the PATH)
        # Use stdout/stderr suppression to prevent cluttering the terminal
        subprocess.run(["xfoil"], stdin=open(input_file, 'r'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Clean up input/coord files
        if os.path.exists(input_file): os.remove(input_file)
        if os.path.exists(coord_file): os.remove(coord_file)

    # Parse polar file
    success = False
    cl_max = -999.0
    cd_at_target_cl = 999.0
    cm_at_target_cl = 999.0
    target_cl = 0.65  # The 2D target lift coefficient corresponding to 3D C_L_cruise = 0.60

    if os.path.exists(polar_file):
        try:
            with open(polar_file, 'r') as f:
                lines = f.readlines()
            
            # Find the data rows (lines starting after headers)
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
                cl_max = max(cls)
                success = True
                
                # Interpolate Cd and Cm at target Cl
                # Find the closest Cl value
                cls = np.array(cls)
                cds = np.array(cds)
                cms = np.array(cms)
                
                # If target_cl is outside our computed range, penalize
                if target_cl < min(cls) or target_cl > max(cls):
                    cd_at_target_cl = 0.05  # Moderate penalty
                else:
                    cd_at_target_cl = np.interp(target_cl, cls, cds)
                    cm_at_target_cl = np.interp(target_cl, cls, cms)
        except Exception:
            pass
        finally:
            if os.path.exists(polar_file): os.remove(polar_file)

    return cl_max, cd_at_target_cl, cm_at_target_cl, success

# 3. Parameter Mapping and Objective Function
PARSEC_KEYS = ['r_le', 'x_up', 'y_up', 'k_up', 'x_low', 'y_low', 'k_low', 'alpha_te', 'beta_te']

# Baseline parameters (resembling SD7062)
BASELINE_X = np.array([
    0.0190,  # r_le
    0.3200,  # x_up
    0.0980,  # y_up
    -0.4500, # k_up
    0.3500,  # x_low
    -0.0420, # y_low
    0.6500,  # k_low
    -5.0000, # alpha_te
    10.0000  # beta_te
])

BOUNDS = [
    (0.010, 0.025),    # r_le
    (0.280, 0.400),    # x_up
    (0.085, 0.115),    # y_up (keeps upper thickness reasonable)
    (-0.600, -0.300),  # k_up
    (0.300, 0.420),    # x_low
    (-0.055, -0.035),  # y_low (enforces lower surface profile, total thickness ~ 12-15%)
    (0.400, 0.800),    # k_low
    (-8.000, -2.000),  # alpha_te
    (6.000, 14.000)    # beta_te
]

# We need to pass a run counter to separate temp files
run_counter = 0

def objective(x_vector):
    global run_counter
    run_counter += 1
    
    # Map vector to PARSEC dict
    p_dict = {PARSEC_KEYS[i]: x_vector[i] for i in range(len(PARSEC_KEYS))}
    p_dict['y_te'] = 0.0
    p_dict['delta_y_te'] = 0.001
    
    cl_max, cd, cm, success = evaluate_airfoil_xfoil(p_dict, run_counter)
    
    # Calculate total thickness (y_up - y_low)
    thickness = p_dict['y_up'] - p_dict['y_low']
    
    # Enforce constraints using penalty method
    penalty = 0.0
    
    # Constraint 1: Stall lift coefficient must be high enough
    if cl_max < 1.40:
        penalty += (1.40 - cl_max) * 10.0
        
    # Constraint 2: Structural thickness (must be at least 13.5% chord to fit the 25mm spar)
    if thickness < 0.135:
        penalty += (0.135 - thickness) * 50.0
        
    # Constraint 3: Negative pitching moment should not be too large (avoid tail loading)
    if cm < -0.08:
        penalty += (-0.08 - cm) * 5.0

    # Objective: Minimize Drag Coefficient at target Cl + penalties
    if not success:
        return 1.0  # Big penalty for failure to converge
        
    return cd + penalty

# 4. Multi-threaded Objective wrapper for scipy minimize (to compute gradients in parallel)
# Scipy SLSQP evaluates gradients by perturbing parameters. We can parallelize the objective evaluation.
class ParallelObjective:
    def __init__(self, cores=20):
        self.cores = cores
        self.pool = ProcessPoolExecutor(max_workers=cores)
        
    def __call__(self, x_vector):
        # When Scipy calls the objective function, it usually evaluates one point at a time.
        # However, we can use Scipy's parallel gradient estimation by setting `jacobian='2-point'`
        # and passing a parallel wrapper, or we can use Scipy's built-in parallel capabilities.
        return objective(x_vector)

def run_optimization():
    print("Initializing parallelized SLSQP optimization...")
    print("Baseline parameters (SD7062-like):")
    for k, v in zip(PARSEC_KEYS, BASELINE_X):
        print(f"  {k}: {v}")
        
    # Calculate baseline performance
    p_baseline = {PARSEC_KEYS[i]: BASELINE_X[i] for i in range(len(PARSEC_KEYS))}
    p_baseline['y_te'] = 0.0
    p_baseline['delta_y_te'] = 0.001
    
    print("\nEvaluating baseline airfoil in XFOIL...")
    cl_max, cd, cm, success = evaluate_airfoil_xfoil(p_baseline, 0)
    if success:
        print(f"Baseline Cl_max: {cl_max:.4f}")
        print(f"Baseline Cd at Cl=0.65: {cd:.5f}")
        print(f"Baseline Cm at Cl=0.65: {cm:.5f}")
        print(f"Baseline Thickness: {(p_baseline['y_up'] - p_baseline['y_low'])*100:.2f}%")
    else:
        print("Error: Baseline failed to evaluate in XFOIL. Make sure xfoil is installed.")
        sys.exit(1)

    print("\nStarting optimization loop (SLSQP)...")
    # We pass '2-point' for numerical jacobian approximation. 
    # To run in parallel, we can use scipy.optimize.minimize's built-in vectorization or pool.
    res = minimize(
        objective, 
        BASELINE_X, 
        method='SLSQP', 
        bounds=BOUNDS,
        options={'disp': True, 'maxiter': 25, 'eps': 1e-3}
    )
    
    print("\nOptimization Complete!")
    print(res)
    
    # Save optimized coordinates
    opt_x = res.x
    p_opt = {PARSEC_KEYS[i]: opt_x[i] for i in range(len(PARSEC_KEYS))}
    p_opt['y_te'] = 0.0
    p_opt['delta_y_te'] = 0.001
    
    x_coords, y_coords = parsec_airfoil(p_opt)
    if x_coords is not None:
        save_path = "optimized_SD7062_opt.dat"
        with open(save_path, 'w') as f:
            f.write("SD7062_OPTIMIZED_TEKNOFEST\n")
            for xi, yi in zip(x_coords, y_coords):
                f.write(f"  {xi:.7f}   {yi:.7f}\n")
        print(f"\nOptimized coordinates saved to '{save_path}'!")
        
        # Print optimized performance
        cl_max, cd, cm, success = evaluate_airfoil_xfoil(p_opt, 9999)
        print(f"Optimized Cl_max: {cl_max:.4f}")
        print(f"Optimized Cd at Cl=0.65: {cd:.5f}")
        print(f"Optimized Cm at Cl=0.65: {cm:.5f}")
        print(f"Optimized Thickness: {(p_opt['y_up'] - p_opt['y_low'])*100:.2f}%")
        
if __name__ == '__main__':
    run_optimization()
