#Initial Instructions

import os
import msvcrt

# ANSI escape code for light blue text
LIGHT_BLUE = "\033[94m"
RESET = "\033[0m"

print(LIGHT_BLUE + "Must Have Python 3.10 and C++ Build Tools Installed" + RESET)
print("1. Download Python 3.10 from the [Microsoft Store](https://www.microsoft.com/store/productId/9PJPW5LDXLZ5).")
print("2. Download C++ Build Tools from [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).")
print("3. Run the installer, select 'Desktop Development with C++', and complete the installation.")
print("4. Now you can proceed with Voiceclone X.")
print("Press any key to continue...")

msvcrt.getch()  # Waits for any key press


#1. Change Command Prompt Execution Policy

import subprocess

# ANSI escape code for light blue text
LIGHT_BLUE = "\033[94m"
RESET = "\033[0m"

def get_execution_policy():
    """Get the current execution policy from PowerShell."""
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-ExecutionPolicy"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting execution policy: {e}")
        return None

def set_execution_policy(policy):
    """Set the execution policy to the specified policy."""
    try:
        subprocess.run(
            ["powershell", "-Command", f"Set-ExecutionPolicy {policy} -Scope CurrentUser -Force"],
            check=True
        )
        print(f"Execution policy set to {policy}.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting execution policy: {e}")

def main():
    # Ask user if they want to continue with the script
    proceed = input(LIGHT_BLUE + "Do you want to change Command Prompt Execution Policy? (Y/N): " + RESET).strip().lower()
    if proceed != 'y':
        print("Operation cancelled.")
        return

    current_policy = get_execution_policy()
    if current_policy is None:
        print("Failed to retrieve the current execution policy.")
        return

    print(f"Current Execution Policy: {current_policy}")

    if current_policy == "RemoteSigned":
        reset_input = input(LIGHT_BLUE + "The current policy is 'RemoteSigned'. Do you want to reset the execution policy to the default 'Restricted'? (Y/N): " + RESET).strip().lower()
        if reset_input == 'y':
            default_policy = "Restricted"
            print(f"Resetting execution policy to {default_policy}...")
            set_execution_policy(default_policy)
            
            # Ask if the user wants to set it back to RemoteSigned
            reset_back_input = input(LIGHT_BLUE + "Do you want to change the execution policy back to 'RemoteSigned'? (Y/N): " + RESET).strip().lower()
            if reset_back_input == 'y':
                print("Changing execution policy back to RemoteSigned...")
                set_execution_policy("RemoteSigned")
            elif reset_back_input == 'n':
                print("Keeping the current execution policy as 'Restricted'.")
            else:
                print("Invalid input. Please enter 'Y' for yes or 'N' for no.")
        elif reset_input == 'n':
            print("Keeping the current execution policy as 'RemoteSigned'.")
        else:
            print("Invalid input. Please enter 'Y' for yes or 'N' for no.")
    else:
        user_input = input(LIGHT_BLUE + "Do you want to change the execution policy to 'RemoteSigned'? (Y/N): " + RESET).strip().lower()
        if user_input == 'y':
            print("Changing execution policy to RemoteSigned...")
            set_execution_policy("RemoteSigned")
        elif user_input == 'n':
            print("Operation cancelled.")
        else:
            print("Invalid input. Please enter 'Y' for yes or 'N' for no.")

if __name__ == "__main__":
    main()

#2. Create a Virtual Environment

import os
import subprocess
import sys
import shutil
import time

# ANSI escape code for light blue text
LIGHT_BLUE = "\033[94m"
RESET = "\033[0m"

def main():
    proceed = input(LIGHT_BLUE + "Do you want to Create a Virtual Environment? (Y/N): " + RESET).strip().lower()
    if proceed != 'y':
        print("Operation cancelled.")
        return

    base_dir = r"C:\Voiceclone X\environments"
    os.makedirs(base_dir, exist_ok=True)

    env_name = input("Enter the name for the virtual environment: ")
    env_path = create_virtual_environment(base_dir, env_name)

    install_packages_prompt = input("Would you like to install the requirements for voice cloning? (Y/N): ").strip().lower()
    if install_packages_prompt == 'y':
        packages = [
            "fairseq==0.12.2",
            "faiss-cpu==1.7.4",
            "numpy==1.23.5",
            "scikit-learn==1.3.0",
            "torch==2.0.0",
            "torchvision==0.15.0",
            "torchaudio==0.12.1",
            "librosa==0.10.0.post2",
            "tensorflow==2.13.0",
            "av==10.0.0",
            "pandas==2.0.3",
            "requests==2.31.0",
            "praat-parselmouth==0.4.3",
            "tqdm==4.65.0",
            "pyworld==0.3.2",
            "mkl-static==2024.2.0",
            "mkl-include==2024.2.0",
            "matplotlib>=3.9.1",
            "matplotlib-inline==0.1.7",
            "pillow==9.5.0"
        ]
        install_packages(env_path, packages)

    print(f"\nVirtual environment created at: {env_path}")

def create_virtual_environment(base_dir, env_name):
    """Create a virtual environment with the specified name."""
    env_path = os.path.join(base_dir, env_name)
    print(f"Creating virtual environment at: {env_path}")
    subprocess.run([sys.executable, "-m", "venv", env_path], check=True)
    return env_path

def install_packages(env_path, packages):
    """Install specified packages into the virtual environment."""
    pip_executable = os.path.join(env_path, 'Scripts', 'pip.exe') if os.name == 'nt' else os.path.join(env_path, 'bin', 'pip')

    total_packages = len(packages)
    if total_packages == 0:
        print("No packages specified.")
        return

    print(f"Total packages to install: {total_packages}")

    for i, package in enumerate(packages):
        print(f"\nInstalling package {i + 1}/{total_packages}: {package}")
        display_loading_bar(0, 100)  # Start the loading bar at 0%

        result = subprocess.run([pip_executable, 'install', package], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully installed {package}.")
        else:
            print(f"Failed to install {package}. Error: {result.stderr}")

        display_loading_bar(100, 100)  # Complete the loading bar

    print("All specified packages have been installed.")
    pip_cache_purge(pip_executable)

def display_loading_bar(progress, total):
    """Display a simple loading bar."""
    bar_length = 40
    completed = int(bar_length * progress / total)
    bar = '#' * completed + '-' * (bar_length - completed)
    print(f"[{bar}] {progress}/{total}%")
    time.sleep(0.1)  # Simulate progress

def pip_cache_purge(pip_executable):
    """Purge the pip cache."""
    result = subprocess.run([pip_executable, 'cache', 'purge'], capture_output=True, text=True)
    if result.returncode == 0:
        print("Pip cache purged successfully.")
    else:
        print(f"Failed to purge pip cache. Error: {result.stderr}")
    
    delete_user_cache()

def delete_user_cache():
    """Delete the user cache folder for pip."""
    user_home = os.path.expanduser('~')
    local_cache_dir = os.path.join(user_home, 'AppData', 'Local', 'Packages')
    if not os.path.exists(local_cache_dir):
        print("Local cache directory does not exist.")
        return

    for root, dirs, files in os.walk(local_cache_dir):
        for dir_name in dirs:
            if "PythonSoftwareFoundation.Python" in dir_name:
                pip_cache_dir = os.path.join(root, dir_name, 'LocalCache', 'Local', 'pip', 'cache')
                if os.path.exists(pip_cache_dir):
                    print(f"Deleting pip cache directory: {pip_cache_dir}")
                    try:
                        shutil.rmtree(pip_cache_dir)
                        print("Pip cache directory deleted successfully.")
                    except Exception as e:
                        print(f"Failed to delete pip cache directory. Error: {e}")
                return

    print("Pip cache directory not found.")

if __name__ == "__main__":
    main()

#3. Activate Virtual Environment

import os
import subprocess
import sys

# ANSI escape codes for colors
LIGHT_BLUE = "\033[94m"
LIGHT_GREEN = "\033[92m"
RESET = "\033[0m"

# Path to the environments directory
ENVIRONMENTS_DIR = r"C:\Voiceclone X\environments"

# Function to list available virtual environments
def list_virtual_environments(environments_dir):
    try:
        environments = [d for d in os.listdir(environments_dir) if os.path.isdir(os.path.join(environments_dir, d))]
        if not environments:
            print("No virtual environments found in the specified directory.")
            sys.exit(1)
        return environments
    except FileNotFoundError:
        print(f"The directory {environments_dir} does not exist.")
        sys.exit(1)

# Prompt user to select a virtual environment
def select_virtual_environment(environments):
    print("Available virtual environments:")
    for idx, env in enumerate(environments, start=1):
        print(f"{idx}. {env}")
    
    choice = input("Enter the number of the virtual environment you want to use: ").strip()
    
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(environments):
            return environments[choice_idx]
        else:
            print("Invalid choice. Please enter a valid number.")
            sys.exit(1)
    except ValueError:
        print("Invalid input. Please enter a number.")
        sys.exit(1)

# Prompt user to continue
def prompt_to_continue():
    response = input(LIGHT_BLUE + "Do you want to activate a virtual environment? (Y/N): " + RESET).strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        sys.exit(0)

def main():
    # Prompt user to continue
    prompt_to_continue()
    
    # List and select virtual environment
    environments = list_virtual_environments(ENVIRONMENTS_DIR)
    selected_env = select_virtual_environment(environments)

    VIRTUAL_ENVIRONMENT_DIR = os.path.join(ENVIRONMENTS_DIR, selected_env)
    activate_script = os.path.join(VIRTUAL_ENVIRONMENT_DIR, "Scripts", "activate.bat")

    # Path to the script to run
    script_to_run = r"C:\Voiceclone X\apps\voiceclonex APP.py"

    # Display selected environment
    print(f"Selected virtual environment: " + LIGHT_GREEN + f"{selected_env}" + RESET)

    # Use subprocess to execute the activation script
    try:
        # Properly quote the paths
        activate_command = f'"{activate_script}"'
        
        # Combined command to activate the environment and run the script
        run_script_command = f'cmd /k "{activate_command} && python "{script_to_run}""'
        
        subprocess.run(run_script_command, shell=True, check=True)
        print(f"Successfully activated the virtual environment '{selected_env}' and ran the script.")
        
    except subprocess.CalledProcessError:
        print(f"Failed to activate the virtual environment '{selected_env}' or run the script.")
        sys.exit(1)

if __name__ == "__main__":
    main()