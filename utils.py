import os
import sys
import subprocess
import platform
import shutil
import logging

def install_dependency(package, console=None):
    """Installs a Python package using pip, with user feedback."""
    msg = f"Attempting to install '{package}' via pip..."
    logging.info(msg)
    if console:
        console.print(f"[cyan]{msg}[/cyan]")
    else:
        print(msg)

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'setuptools'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        if console:
            console.print(f"[bold green]Successfully installed '{package}'.[/bold green]")
        else:
            print(f"Successfully installed '{package}'.")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"Error installing '{package}': {e}"
        logging.error(error_msg)
        if console:
            console.print(f"[bold red]{error_msg}[/bold red]")
            console.print(f"[yellow]Please try installing '{package}' manually.[/yellow]")
        else:
            print(error_msg)
            print(f"Please try installing '{package}' manually.")
        return False

def install_nodejs_and_peerjs(console=None):
    """Installs Node.js and all necessary dependencies for the PeerJS bridge."""
    if platform.system() != "Linux":
        if console:
            console.print("[red]Automatic dependency installation is only supported on Linux.[/red]")
        else:
            print("Automatic dependency installation is only supported on Linux.")
        return False

    # Install Node.js if not present
    if not shutil.which('node'):
        if console:
            console.print("[yellow]Node.js not found. Attempting to install...[/yellow]")
        else:
            print("Node.js not found. Attempting to install...")
        try:
            subprocess.check_call("sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q nodejs npm", shell=True)
            if console:
                console.print("[green]Node.js installed successfully.[/green]")
            else:
                print("Node.js installed successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if console:
                console.print(f"[red]Failed to install Node.js: {e}[/red]")
            else:
                print(f"Failed to install Node.js: {e}")
            return False
    else:
        if console:
            console.print("[green]Node.js is already installed.[/green]")
        else:
            print("Node.js is already installed.")

    # Install system packages required for Electron/PeerJS
    system_packages = ['xvfb', 'libgtk2.0-0', 'libdbus-glib-1-2']
    try:
        package_str = " ".join(system_packages)
        if console:
            console.print(f"[yellow]Installing required system packages: {package_str}...[/yellow]")
        else:
            print(f"Installing required system packages: {package_str}...")
        subprocess.check_call(f"sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q {package_str}", shell=True)
        if console:
            console.print(f"[green]System packages ({package_str}) installed successfully.[/green]")
        else:
            print(f"System packages ({package_str}) installed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if console:
            console.print(f"[red]Failed to install system packages: {e}[/red]")
        else:
            print(f"Failed to install system packages: {e}")
        return False

    # Manually install legacy gconf packages
    try:
        if console:
            console.print("[yellow]Manually installing legacy gconf packages...[/yellow]")
        else:
            print("Manually installing legacy gconf packages...")

        # Download and install gconf2-common first
        subprocess.check_call("wget -q http://archive.ubuntu.com/ubuntu/pool/universe/g/gconf/gconf2-common_3.2.6-7ubuntu2_all.deb", shell=True)
        subprocess.check_call("sudo dpkg -i gconf2-common_3.2.6-7ubuntu2_all.deb", shell=True)

        # Download and install libgconf-2-4
        subprocess.check_call("wget -q http://archive.ubuntu.com/ubuntu/pool/universe/g/gconf/libgconf-2-4_3.2.6-7ubuntu2_amd64.deb", shell=True)
        subprocess.check_call("sudo dpkg -i libgconf-2-4_3.2.6-7ubuntu2_amd64.deb", shell=True)

        # Fix any broken dependencies
        subprocess.check_call("sudo apt-get -f install -y", shell=True)

        # Clean up downloaded files
        subprocess.check_call("rm gconf2-common_3.2.6-7ubuntu2_all.deb libgconf-2-4_3.2.6-7ubuntu2_amd64.deb", shell=True)

        if console:
            console.print("[green]Legacy gconf packages installed successfully.[/green]")
        else:
            print("Legacy gconf packages installed successfully.")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if console:
            console.print(f"[red]Failed to install legacy gconf packages: {e}[/red]")
        else:
            print(f"Failed to install legacy gconf packages: {e}")
        return False

    # Install local npm packages
    if console:
        console.print("[cyan]Installing local Node.js dependencies...[/cyan]")
    else:
        print("Installing local Node.js dependencies...")
    try:
        subprocess.check_call("npm install", shell=True)
        if console:
            console.print("[green]Node.js dependencies installed successfully.[/green]")
        else:
            print("Node.js dependencies installed successfully.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if console:
            console.print(f"[red]Failed to install Node.js dependencies: {e}[/red]")
            if hasattr(e, 'stderr') and e.stderr:
                console.print(f"[red]Stderr: {e.stderr}[/red]")
        else:
            print(f"Failed to install Node.js dependencies: {e}")
        return False