import subprocess
import sys

with open('requirements.txt', 'r') as f:
    packages = f.readlines()

for package in packages:
    package = package.strip()
    if not package or package.startswith('#'):
        continue
    print(f"--- Installing {package} ---")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"--- Successfully installed {package} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- FAILED to install {package}: {e} ---")

print("--- Installation attempt finished. ---")
