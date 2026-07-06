import os
import sys
import urllib.request
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt

def run_polar(coord_file, name, run_id, Re=460000):
    polar_file = f"temp_comp_polar_{run_id}.txt"
    input_file = f"temp_comp_input_{run_id}.in"
    
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
        
    cmd = ["xvfb-run", "-a", "xfoil"] if shutil.which("xvfb-run") else ["xfoil"]
    subprocess.run(cmd, stdin=open(input_file, 'r'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    
    if os.path.exists(input_file): os.remove(input_file)
    
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
    return np.array(alphas), np.array(cls), np.array(cds), np.array(cms)

def main():
    print("Generating comparative plots for baseline SD7062 vs Optimized Perfect Airfoil at Re=460,000...")
    
    # 1. Get baseline SD7062
    url = "https://m-selig.ae.illinois.edu/ads/coord/sd7062.dat"
    baseline_file = "sd7062_baseline_tmp.dat"
    urllib.request.urlretrieve(url, baseline_file)
    
    # Read baseline coordinates
    with open(baseline_file, 'r') as f:
        lines = f.readlines()
    baseline_coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            try: baseline_coords.append([float(parts[0]), float(parts[1])])
            except ValueError: continue
    baseline_coords = np.array(baseline_coords)
    
    # 2. Get optimized coordinates
    opt_file = "../optimized_SD7062_perfect.dat"
    with open(opt_file, 'r') as f:
        lines = f.readlines()
    opt_coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            try: opt_coords.append([float(parts[0]), float(parts[1])])
            except ValueError: continue
    opt_coords = np.array(opt_coords)
    
    # Plot geometry comparison
    plt.figure(figsize=(10, 3.5))
    plt.plot(baseline_coords[:, 0], baseline_coords[:, 1], 'b--', linewidth=2, label="SD7062 Baseline")
    plt.plot(opt_coords[:, 0], opt_coords[:, 1], 'r-', linewidth=2, label="SD7062 Robust Optimized (7.5kg)")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil Geometry Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("airfoil_comparison.png", dpi=300)
    plt.close()
    print("Saved airfoil_comparison.png")
    
    # Run polars
    alp_b, cl_b, cd_b, cm_b = run_polar(baseline_file, "Baseline", 1)
    alp_o, cl_o, cd_o, cm_o = run_polar(opt_file, "Optimized", 2)
    
    if os.path.exists(baseline_file): os.remove(baseline_file)
    
    # Plot polars
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cl vs alpha
    axs[0, 0].plot(alp_b, cl_b, 'b--', linewidth=2, label="SD7062 Baseline")
    axs[0, 0].plot(alp_o, cl_o, 'r-', linewidth=2, label="SD7062 Robust Optimized")
    axs[0, 0].set_xlabel("alpha (deg)")
    axs[0, 0].set_ylabel("Cl")
    axs[0, 0].set_title("Lift Coefficient vs. Angle of Attack")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 2. Cl vs Cd
    axs[0, 1].plot(cd_b, cl_b, 'b--', linewidth=2)
    axs[0, 1].plot(cd_o, cl_o, 'r-', linewidth=2)
    axs[0, 1].set_xlabel("Cd")
    axs[0, 1].set_ylabel("Cl")
    axs[0, 1].set_title("Drag Polar")
    axs[0, 1].grid(True)
    
    # 3. Cm vs alpha
    axs[1, 0].plot(alp_b, cm_b, 'b--', linewidth=2)
    axs[1, 0].plot(alp_o, cm_o, 'r-', linewidth=2)
    axs[1, 0].set_xlabel("alpha (deg)")
    axs[1, 0].set_ylabel("Cm")
    axs[1, 0].set_title("Pitching Moment Coefficient vs. Angle of Attack")
    axs[1, 0].grid(True)
    
    # 4. L/D vs alpha
    axs[1, 1].plot(alp_b, cl_b/cd_b, 'b--', linewidth=2)
    axs[1, 1].plot(alp_o, cl_o/cd_o, 'r-', linewidth=2)
    axs[1, 1].set_xlabel("alpha (deg)")
    axs[1, 1].set_ylabel("Cl/Cd")
    axs[1, 1].set_title("Glide Efficiency (Cl/Cd) vs. Angle of Attack")
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("polar_comparison.png", dpi=300)
    plt.close()
    print("Saved polar_comparison.png")

if __name__ == '__main__':
    main()
