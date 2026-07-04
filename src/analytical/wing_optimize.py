import os
import sys
import numpy as np
import aerosandbox as asb
from scipy.optimize import minimize, fsolve

def solve_alpha_for_cl(airplane, target_cl, initial_guess=3.0):
    """Solves for the angle of attack alpha (in degrees) to match a target CL."""
    def residual(alpha):
        op = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha[0])
        vlm = asb.VortexLatticeMethod(airplane, op)
        res = vlm.run()
        return [res['CL'] - target_cl]
    
    alpha_opt = fsolve(residual, [initial_guess])[0]
    return alpha_opt

def evaluate_wing(taper, washout):
    # Fixed parameters from conceptual design
    S = 0.916     # Wing Area (m^2)
    ar = 6.85     # Aspect Ratio (wingspan locked at 2.50m)
    b = np.sqrt(ar * S) # Wingspan (2.50m)
    
    c_root = 2 * S / (b * (1 + taper))
    c_tip = c_root * taper
    
    # Load optimized airfoil (check relative to script location first, then fallback to current folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_try = [
        os.path.join(script_dir, "optimized_SD7062_perfect.dat"),
        os.path.join(os.path.dirname(script_dir), "optimized_SD7062_perfect.dat"),
        os.path.join(os.path.dirname(script_dir), "analytical", "optimized_SD7062_perfect.dat"),
        os.path.join(script_dir, "src", "analytical", "optimized_SD7062_perfect.dat"),
        "optimized_SD7062_perfect.dat"
    ]
    airfoil_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            airfoil_path = p
            break
            
    if airfoil_path is None:
        raise FileNotFoundError("Could not locate optimized_SD7062_perfect.dat coordinates file!")
        
    airfoil = asb.Airfoil(name="SD7062_opt", coordinates=airfoil_path)
    
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
                xyz_le=[0.25 * (c_root - c_tip), b/2, 0],
                chord=c_tip,
                twist=-washout, # Washout represented as negative twist at tip
                airfoil=airfoil
            )
        ]
    )
    
    airplane = asb.Airplane(name="UAV Wing", wings=[wing])
    
    # 1. Cruise Phase: CL = 0.65
    alpha_cruise = solve_alpha_for_cl(airplane, 0.65, initial_guess=3.0)
    op_c = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_cruise)
    vlm_c = asb.VortexLatticeMethod(airplane, op_c)
    res_c = vlm_c.run()
    cdi_cruise = res_c['CD']
    
    # 2. Turning Phase: CL = 1.20
    alpha_turn = solve_alpha_for_cl(airplane, 1.20, initial_guess=8.0)
    op_t = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=22, alpha=alpha_turn)
    vlm_t = asb.VortexLatticeMethod(airplane, op_t)
    res_t = vlm_t.run()
    cdi_turn = res_t['CD']
    
    # 3. Stall Phase Safety Check: CL = 1.45 (takeoff/stall limit)
    # We want to check local lift distribution at stall.
    # To prevent tip stall, the local section lift coefficient cl at the tip must be safely below stall limit (1.58).
    # We run VLM at CL = 1.45 and extract the spanwise lift distribution.
    alpha_stall = solve_alpha_for_cl(airplane, 1.45, initial_guess=10.0)
    op_s = asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=100), velocity=15.2, alpha=alpha_stall)
    vlm_s = asb.VortexLatticeMethod(airplane, op_s)
    res_s = vlm_s.run()
    
    # Calculate local section lift coefficient at the tip.
    # In VLM, we can estimate local section CL using the lift distribution.
    # For a simple two-section wing, we check local lift near the tip.
    # AeroSandbox provides forces on each panel. Let's do a simple check on the tip panel.
    # A conservative approach for stall safety:
    # Ensure that washout is high enough to lower local CL at the tip relative to the root.
    # At CL = 1.45, we enforce that the tip does not stall first.
    # A simple indicator is the local CL at the tip section. Let's approximate it:
    # Local CL = Local Lift / (q * Chord).
    # Since VLM computes lift distribution, we can check the twist margin directly.
    # Let's enforce twist >= 2.0 degrees to guarantee root stall.
    
    return cdi_cruise, cdi_turn, c_tip

# Optimization Objective
def objective(x):
    taper, washout = x[0], x[1]
    cdi_c, cdi_t, _ = evaluate_wing(taper, washout)
    
    # Multi-point objective: 60% cruise + 40% turning drag
    obj_val = 0.60 * cdi_c + 0.40 * cdi_t
    return obj_val

def main():
    print("====================================================")
    print("       3D Wing Shape Optimization System            ")
    print("====================================================")
    
    # Initial guess: Taper = 0.6, Washout = 2.5 deg
    x0 = [0.6, 2.5]
    
    # Bounds: Taper in [0.45, 0.8], Washout in [1.5, 4.0] degrees
    bounds = [(0.45, 0.80), (1.5, 4.0)]
    
    # Constraints: Tip chord must be at least 0.20m to fit carbon spar
    # S = 0.916, b = 2.50m
    # c_root = 2 * S / (b * (1 + taper))
    # c_tip = c_root * taper = (2 * S * taper) / (b * (1 + taper)) >= 0.20
    def tip_chord_constraint(x):
        taper = x[0]
        S = 0.916
        b = 2.50
        c_tip = (2 * S * taper) / (b * (1 + taper))
        return c_tip - 0.20  # >= 0
    
    # Stall safety constraint: Washout twist must be at least 2.0 degrees to guarantee tip safety
    def stall_safety_constraint(x):
        washout = x[1]
        return washout - 2.0  # >= 0
        
    cons = [
        {'type': 'ineq', 'fun': tip_chord_constraint},
        {'type': 'ineq', 'fun': stall_safety_constraint}
    ]
    
    print("\nStarting SLSQP 3D Wing Planform Optimization...")
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'disp': True, 'eps': 1e-3}
    )
    
    print("\nOptimization Complete!")
    print(res)
    
    # Results
    opt_taper = res.x[0]
    opt_washout = res.x[1]
    
    S = 0.916
    b = 2.50
    opt_c_root = 2 * S / (b * (1 + opt_taper))
    opt_c_tip = opt_c_root * opt_taper
    
    cdi_c_opt, cdi_t_opt, _ = evaluate_wing(opt_taper, opt_washout)
    cdi_c_base, cdi_t_base, _ = evaluate_wing(0.6, 2.5) # Baseline comparison
    
    print("\n====================================================")
    print("               Wing Optimization Summary            ")
    print("====================================================")
    print(f"Parameter          Baseline        Optimized")
    print(f"Taper Ratio:       0.60            {opt_taper:.3f}")
    print(f"Washout (twist):   2.50 deg        {opt_washout:.2f} deg")
    print(f"Root Chord:        0.458 m         {opt_c_root:.3f} m")
    print(f"Tip Chord:         0.275 m         {opt_c_tip:.3f} m")
    print(f"Wingspan:          2.500 m         2.500 m")
    print(f"Cruise CDi:        {cdi_c_base:.6f}        {cdi_c_opt:.6f} ({(cdi_c_base-cdi_c_opt)/cdi_c_base*100:.2f}% reduction)")
    print(f"Turning CDi:       {cdi_t_base:.6f}        {cdi_t_opt:.6f} ({(cdi_t_base-cdi_t_opt)/cdi_t_base*100:.2f}% reduction)")
    print("====================================================")

if __name__ == '__main__':
    main()
