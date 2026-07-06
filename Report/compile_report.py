import os
import subprocess
import sys

def compile_latex():
    tex_files = [
        "airfoil_optimization_report.tex",
        "wing_optimization_report.tex",
        "stability_and_empennage_report.tex"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    for tex_file in tex_files:
        pdf_file = tex_file.replace(".tex", ".pdf")
        print(f"\nCompiling {tex_file} in directory: {script_dir}...")
        
        if not os.path.exists(tex_file):
            print(f"Error: {tex_file} not found in {script_dir}!")
            sys.exit(1)
            
        try:
            print("Running pdflatex (Pass 1)...")
            result1 = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if result1.returncode != 0:
                print("Error: LaTeX compilation failed in first pass.")
                print(result1.stdout.decode('utf-8', errors='ignore')[-2000:])
                sys.exit(1)
                
            print("Running pdflatex (Pass 2)...")
            result2 = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if result2.returncode == 0:
                print(f"Success! Compiled PDF saved to: {os.path.join(script_dir, pdf_file)}")
                
                # Clean up auxiliary files
                aux_extensions = ['.aux', '.log', '.out', '.toc']
                for ext in aux_extensions:
                    file_to_clean = tex_file.replace('.tex', ext)
                    if os.path.exists(file_to_clean):
                        os.remove(file_to_clean)
                print(f"Cleaned up auxiliary files for {tex_file}.")
            else:
                print("Error: LaTeX compilation failed in second pass.")
                print(result2.stdout.decode('utf-8', errors='ignore')[-2000:])
                sys.exit(1)
                
        except FileNotFoundError:
            print("Error: 'pdflatex' command not found! Make sure a LaTeX distribution is installed and in your PATH.")
            sys.exit(1)

if __name__ == '__main__':
    compile_latex()
