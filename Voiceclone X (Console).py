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

#1. Create a Virtual Environment

import os
import subprocess
import sys
import shutil
import time

def main():
    proceed = input("Do you want to Create a Virtual Environment? (Y/N): ").strip().lower()
    if proceed != 'y':
        print("Operation cancelled.")
        return

    base_dir = r"C:\Voiceclone X\environments"
    os.makedirs(base_dir, exist_ok=True)

    env_name = input("Enter the name for the virtual environment: ")
    env_path = create_virtual_environment(base_dir, env_name)

    install_packages_prompt = input("would you like to install the requirements for voice cloning? (Y/N): ").strip().lower()
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

#2. Activate Virtual Environment

import os
import subprocess
import sys

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
    response = input("Do you want to activate a virtual environment? (Y/N): ").strip().lower()
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
    print(f"Selected virtual environment: {selected_env}")

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

#3. Install Project Dependency  
 
#Ask the user if they want to download and continue
user_input = input("Install Project Dependencies? (Y/N): ").strip().lower()

if user_input != 'y':
    print("Operation cancelled.")
else:
    import os
    import requests
    from zipfile import ZipFile
    from tqdm import tqdm
    import shutil

    # URL and local paths
    zip_url = "https://huggingface.co/datasets/lilbotai/RVC_Installer/resolve/main/RVC.zip?download=true"
    zip_file_path = r"C:\RVC\RVC.zip"
    extracted_folder_path = r"C:\RVC"

    # Create the directory if it doesn't exist
    os.makedirs(extracted_folder_path, exist_ok=True)

    # Download the zip file with a loading circle
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(zip_file_path, "wb") as zip_file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            zip_file.write(data)

    # Extract the contents
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)

    # Move the 'content' folder to the main 'Voice Training' folder
    content_folder_path = os.path.join(extracted_folder_path, 'RVC', 'content')
    shutil.move(content_folder_path, extracted_folder_path)

    # Delete the empty 'RVC' folder
    rvc_folder_path = os.path.join(extracted_folder_path, 'RVC')
    os.rmdir(rvc_folder_path)

    # Delete the zip file
    os.remove(zip_file_path)

    print("Installation complete")

#4. Create a Dataset

import os
import shutil
import wave
import math

def get_user_input(prompt, default_value=None, input_type=str):
    user_input = input(f"{prompt} [{default_value}]: ").strip()
    return input_type(user_input) if user_input else default_value

def segment_wav(input_file, output_folder, segment_duration=10):
    with wave.open(input_file, 'rb') as input_wav:
        sample_width = input_wav.getsampwidth()
        framerate = input_wav.getframerate()
        num_frames = input_wav.getnframes()
        total_duration = num_frames / framerate
        frames_per_segment = int(framerate * segment_duration)
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]

        os.makedirs(output_folder, exist_ok=True)

        for segment_index in range(math.ceil(total_duration / segment_duration)):
            start_frame = segment_index * frames_per_segment
            end_frame = min((segment_index + 1) * frames_per_segment, num_frames)

            input_wav.setpos(start_frame)
            frames = input_wav.readframes(end_frame - start_frame)

            output_file = os.path.join(output_folder, f"{input_file_name}_segment{segment_index + 1}.wav")
            with wave.open(output_file, 'wb') as output_wav:
                output_wav.setnchannels(input_wav.getnchannels())
                output_wav.setsampwidth(sample_width)
                output_wav.setframerate(framerate)
                output_wav.writeframes(frames)

            print(f"Segment {segment_index + 1} created: {output_file}")

def main():
    user_input = input("Dataset? (Y/N): ").strip().lower()
    
    if user_input != 'y':
        print("Operation cancelled.")
        return

    input_file_path = input("Enter the path to the audio WAV file: ").strip('"')
    output_folder_path = rf"C:\RVC\content\dataset\{os.path.splitext(os.path.basename(input_file_path))[0]}"
    segment_duration_seconds = get_user_input("Enter the segment duration in seconds (10 seconds is the limit)", 10, float)

    segment_wav(input_file_path, output_folder_path, segment_duration_seconds)
    
    # Copy the voice audio to the specified path
    saved_audios_path = r"C:\RVC\content\saved_audios"
    os.makedirs(saved_audios_path, exist_ok=True)
    shutil.copy2(input_file_path, saved_audios_path)
    print(f"Saved {input_file_path} to {saved_audios_path}")

if __name__ == "__main__":
    main()

#5. Preprocess Data

import os

# Ask the user if they want to preprocess their data and continue
user_input = input("Preprocess Data? (Y/N): ").strip().lower()

if user_input != 'y':
    print("Operation cancelled.")
else:
    # Data Preparation
    os.chdir(r'C:\RVC\content\project-main')
    dataset_folder = r'C:\RVC\content\dataset'  # Dataset folder path

    # Get the list of models (subdirectories) in the dataset folder
    models = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]

    if not models:
        print("No models found in the dataset folder.")
    else:
        # Display dropdown menu for model selection
        print("Select a dataset from the list below:")
        for i, model in enumerate(models, start=1):
            print(f"{i}. {model}")
        
        model_choice = int(input("Enter the number corresponding to your model: "))
        model_name = models[model_choice - 1]  # Get the selected model name

        # Path to the specific model's dataset folder
        model_dataset_folder = os.path.join(dataset_folder, model_name)

        while len(os.listdir(model_dataset_folder)) < 1:
            input("Your dataset folder is empty. Please make sure it contains the necessary audio files.")

        # Create logs directory
        os.makedirs(os.path.join('.', 'logs', model_name), exist_ok=True)

        # Preprocessing
        preprocess_log_path = os.path.join('.', 'logs', model_name, 'preprocess.log')
        with open(preprocess_log_path, 'w') as f:
            print("Starting...")

        # Execute preprocessing script
        os.system(f'python infer/modules/train/preprocess.py {model_dataset_folder} 40000 2 ./logs/{model_name} False 3.0 > nul 2>&1')

        # Check if preprocessing was successful
        with open(preprocess_log_path, 'r') as f:
            if 'end preprocess' in f.read():
                print("\u2714 Success")
            else:
                print("Error preprocessing data. Please ensure your dataset folder is correct.")

#6. Extract Features

import os

# Ask the user if they want to extract features and continue
user_input = input("Extract Features? (Y/N): ").strip().lower()

if user_input == 'y':
    # Define the main path
    main_path = r'C:\RVC\content\project-main'

    # Get the list of logs (subdirectories) in the logs folder
    logs_folder = os.path.join(main_path, 'logs')
    models = [d for d in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, d))]

    if not models:
        print("No models found in the logs folder.")
    else:
        # Display dropdown menu for log folder selection
        print("Select a model folder from the list below:")
        for i, model in enumerate(models, start=1):
            print(f"{i}. {model}")

        model_choice = int(input("Enter the number corresponding to your model: "))
        model_name = models[model_choice - 1]  # Get the selected model name

        # Define the dataset folder path based on the model_name input
        dataset_path = os.path.join(r'C:\RVC\content\dataset', model_name)

        # Present options for f0method
        f0methods = ['pm', 'harvest', 'rmvpe', 'rmvpe_gpu']
        print("Choose f0method:")
        for i, method in enumerate(f0methods, 1):
            print(f"{i}. {method}")

        # Input function for f0method
        choice = input("Enter the number corresponding to your choice: ").strip()

        try:
            choice = int(choice)
            if 1 <= choice <= len(f0methods):
                f0method = f0methods[choice - 1]
            else:
                raise ValueError("Invalid choice.")
        except ValueError as e:
            print(f"Error: {e}")
            print("Invalid input. Please restart the script and choose a valid number.")
            exit()

        # Change directory to the main path
        os.chdir(main_path)

        # Create log directory if not exists
        log_dir = os.path.join(main_path, 'logs', model_name)
        os.makedirs(log_dir, exist_ok=True)

        # Open log file
        with open(os.path.join(log_dir, 'extract_f0_feature.log'), 'w') as f:
            print("Starting...")

        # Run the appropriate command based on f0method
        if f0method != "rmvpe_gpu":
            os.system(r'python infer\modules\train\extract\extract_f0_print.py {} 2 {}'.format(log_dir, f0method))
        else:
            os.system(r'python infer\modules\train\extract\extract_f0_rmvpe.py 1 0 0 {} True'.format(log_dir))

        # Extract feature
        os.system(r'python infer\modules\train\extract_feature_print.py cuda:0 1 0 0 {} v2'.format(log_dir))

        # Check if extraction is successful
        with open(os.path.join(log_dir, 'extract_f0_feature.log'), 'r') as f:
            if 'all-feature-done' in f.read():
                print("\u2714 Success")
            else:
                print("Error preprocessing data... Make sure your data was preprocessed.")
else:
    print("Operation cancelled.")

#7. Train Index

import os
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
import traceback

# Ask the user if they want to train their index and continue
user_input = input("Train Index? (Y/N): ").strip().lower()

if user_input != 'y':
    print("Operation cancelled.")
else:
    def train_index(model_name):
        main_path = r"C:\RVC\content\project-main"
        exp_dir = os.path.join(main_path, "logs", model_name)
        os.makedirs(exp_dir, exist_ok=True)

        feature_dir = os.path.join(exp_dir, "3_feature256" if version19 == "v1" else "3_feature768")

        if not os.path.exists(feature_dir):
            return "Please perform feature extraction first!"

        listdir_res = list(os.listdir(feature_dir))

        if len(listdir_res) == 0:
            return "Please perform feature extraction first！"

        infos = []
        npys = []

        for name in sorted(listdir_res):
            phone = np.load(os.path.join(feature_dir, name))
            npys.append(phone)

        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        if big_npy.shape[0] > 2e5:
            infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
            yield "\n".join(infos)
            try:
                big_npy = (
                    MiniBatchKMeans(
                        n_clusters=10000,
                        verbose=True,
                        batch_size=256 * os.cpu_count(),
                        compute_labels=False,
                        init="random",
                    )
                    .fit(big_npy)
                    .cluster_centers_
                )
            except:
                info = traceback.format_exc()
                infos.append(info)
                yield "\n".join(infos)

        np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        infos.append("%s,%s" % (big_npy.shape, n_ivf))
        yield "\n".join(infos)
        index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
        infos.append("training")
        yield "\n".join(infos)
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)
        faiss.write_index(
            index,
            os.path.join(
                exp_dir,
                "trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (n_ivf, index_ivf.nprobe, model_name, version19),
            ),
        )

        infos.append("adding")
        yield "\n".join(infos)
        batch_size_add = 8192

        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])

        faiss.write_index(
            index,
            os.path.join(
                exp_dir,
                "added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (n_ivf, index_ivf.nprobe, model_name, version19),
            ),
        )

        infos.append(
            "Index built successfully，added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (n_ivf, index_ivf.nprobe, model_name, version19)
        )

    # Define the main path
    main_path = r'C:\RVC\content\project-main'

    # Get the list of logs (subdirectories) in the logs folder
    logs_folder = os.path.join(main_path, 'logs')
    models = [d for d in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, d))]

    if not models:
        print("No models found in the logs folder.")
    else:
        # Display dropdown menu for log folder selection
        print("Select a model folder from the list below:")
        for i, model in enumerate(models, start=1):
            print(f"{i}. {model}")

        model_choice = int(input("Enter the number corresponding to your model: "))
        model_name = models[model_choice - 1]  # Get the selected model name

        version19 = 'v2'  # Assuming this value is fixed
        training_log = train_index(model_name)

        for line in training_log:
            print(line)
            if 'adding' in line:
                print("\u2714 Success")

#8. Train Model 

import os
import pathlib
import json
from subprocess import Popen, PIPE, STDOUT
from random import shuffle
import zipfile

# Function to save the model as a zip file
def save_model_as_zip(model_name):
    # Define main paths
    main_path = r"C:\RVC\content\project-main"
    saved_models_path = r"C:\RVC\content\saved_models"
    
    # Define file paths
    index_file_dir = os.path.join(main_path, "logs", model_name)
    # Automatically detect model name from index file name
    index_file_source = None
    for filename in os.listdir(index_file_dir):
        if filename.endswith(".index") and f"_{model_name}_v2" in filename:
            index_file_source = os.path.join(index_file_dir, filename)
            break
    
    if index_file_source is None:
        print(f"Error: Index file for model '{model_name}' not found.")
        return
    
    weights_file_source = os.path.join(main_path, "assets", "weights", f"{model_name}.pth")
    zip_file_path = os.path.join(saved_models_path, f"{model_name}.zip")

    print(f"Index file path: {index_file_source}")
    print(f"Weights file path: {weights_file_source}")

    # Check if weights file exists
    if not os.path.isfile(weights_file_source):
        print(f"Error: Weights file '{weights_file_source}' not found.")
        return

    # Create zip file for the model
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # Add index file
        zipf.write(index_file_source, os.path.basename(index_file_source))
        # Add weights file
        zipf.write(weights_file_source, os.path.basename(weights_file_source))

    print(f"Model '{model_name}' saved successfully as '{model_name}.zip'.")

# Function to list available models
def list_available_models():
    logs_path = r"C:\RVC\content\project-main\logs"
    models = [d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d)) and d != 'mute']
    return models

# Function to select model
def select_model():
    models = list_available_models()
    if not models:
        print("No models found in the logs directory.")
        return None
    
    print("Available models for training:")
    for idx, model in enumerate(models):
        print(f"{idx + 1}: {model}")

    while True:
        try:
            choice = int(input("Select a model by number: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Ask the user if they want to train their model and continue
user_input = input("Train Model (Y/N): ").strip().lower()

if user_input != 'y':
    print("Operation cancelled.")
else:
    # Select model
    model_name = select_model()
    if model_name is None:
        print("No model selected. Operation cancelled.")
    else:
        # Define main path
        main_path = r'C:\RVC\content\project-main'
        os.chdir(main_path)
        now_dir = os.getcwd()

        # Input functions for user input
        save_frequency = int(input("Choose how often to save the model (1-50): "))
        epochs = int(input("Enter the number of epochs (10,000 Limit): "))
        batch_size = int(input("Enter the batch size (7-32): "))
        gpu_count = int(input("Enter the number of GPUs (0-16): "))

        # Default cache value
        cache = False

        # Automatic search and access functionality
        model_folder_path = os.path.join(now_dir, 'logs', model_name)
        G_path = os.path.join(model_folder_path, 'G_2333333.pth')
        D_path = os.path.join(model_folder_path, 'D_2333333.pth')

        if os.path.exists(G_path) and os.path.exists(D_path):
            pretrained_G = G_path
            pretrained_D = D_path
            print("Resuming training.")
        else:
            print("Files G_2333333.pth and D_2333333.pth not found. Using default paths.")
            pretrained_G = 'assets/pretrained_v2/f0G40k.pth'
            pretrained_D = 'assets/pretrained_v2/f0D40k.pth'

        # Define function to train model
        def click_train(
            exp_dir1,
            sr2,
            if_f0_3,
            spk_id5,
            save_epoch10,
            total_epoch11,
            batch_size12,
            if_save_latest13,
            pretrained_G14,
            pretrained_D15,
            gpus16,
            if_cache_gpu17,
            if_save_every_weights18,
            version19,
        ):
            # Generate filelist
            exp_dir = os.path.join(now_dir, rf'logs\{exp_dir1}')
            os.makedirs(exp_dir, exist_ok=True)
            gt_wavs_dir = os.path.join(exp_dir, '0_gt_wavs')
            feature_dir = os.path.join(exp_dir, '3_feature768') if version19 == "v2" else os.path.join(exp_dir, '3_feature256')
            if if_f0_3:
                f0_dir = os.path.join(exp_dir, '2a_f0')
                f0nsf_dir = os.path.join(exp_dir, '2b-f0nsf')
                names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & \
                        set([name.split(".")[0] for name in os.listdir(feature_dir)]) & \
                        set([name.split(".")[0] for name in os.listdir(f0_dir)]) & \
                        set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
            else:
                names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & \
                        set([name.split(".")[0] for name in os.listdir(feature_dir)])
            opt = []
            for name in names:
                if if_f0_3:
                    opt.append(
                        f"{gt_wavs_dir}\\{name}.wav|{feature_dir}\\{name}.npy|{f0_dir}\\{name}.wav.npy|{f0nsf_dir}\\{name}.wav.npy|{spk_id5}"
                    )
                else:
                    opt.append(
                        f"{gt_wavs_dir}\\{name}.wav|{feature_dir}\\{name}.npy|{spk_id5}"
                    )
            fea_dim = 256 if version19 == "v1" else 768
            if if_f0_3:
                for _ in range(2):
                    opt.append(
                        rf"{now_dir}\logs\mute\0_gt_wavs\mute{sr2}.wav|{now_dir}\logs\mute\3_feature{fea_dim}\mute.npy|{now_dir}\logs\mute\2a_f0\mute.wav.npy|{now_dir}\logs\mute\2b-f0nsf\mute.wav.npy|{spk_id5}"
                    )
            else:
                for _ in range(2):
                    opt.append(
                        rf"{now_dir}\logs\mute\0_gt_wavs\mute{sr2}.wav|{now_dir}\logs\mute\3_feature{fea_dim}\mute.npy|{spk_id5}"
                    )
            shuffle(opt)
            with open(f"{exp_dir}\\filelist.txt", "w") as f:
                f.write("\n".join(opt))

            print("Write filelist done")
            print("Use gpus:", str(gpus16))
            if pretrained_G14 == "":
                print("No pretrained Generator")
            if pretrained_D15 == "":
                print("No pretrained Discriminator")
            if version19 == "v1" or sr2 == "40k":
                config_path = f"configs/v1/{sr2}.json"
            else:
                config_path = f"configs/v2/{sr2}.json"
            config_save_path = os.path.join(exp_dir, "config.json")
            if not pathlib.Path(config_save_path).exists():
                with open(config_save_path, "w", encoding="utf-8") as f:
                    with open(config_path, "r") as config_file:
                        config_data = json.load(config_file)
                        json.dump(
                            config_data,
                            f,
                            ensure_ascii=False,
                            indent=4,
                            sort_keys=True,
                        )
                    f.write("\n")

            cmd = (
                rf'python "C:\RVC\content\project-main\infer\modules\train\train.py" '
                rf'-se {save_epoch10} '
                rf'-te {total_epoch11} '
                rf'-pg {"%s" % pretrained_G14 if pretrained_G14 != "" else ""} '
                rf'-pd {"%s" % pretrained_D15 if pretrained_D15 != "" else ""} '
                rf'-g {gpus16} '
                rf'-bs {batch_size12} '
                rf'-e "{exp_dir1}" '
                rf'-sr {sr2} '
                rf'-sw {1 if if_save_every_weights18 else 0} '
                rf'-v {version19} '
                rf'-f0 {1 if if_f0_3 else 0} '
                rf'-l {1 if if_save_latest13 else 0} '
                
                rf'-c {1 if if_cache_gpu17 else 0}'
            )

            # Execute the command
            p = Popen(cmd, shell=True, cwd=now_dir, stdout=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)

            # Print the output
            for line in p.stdout:
                print(line.strip())

            p.wait()
            return exp_dir1  # Return the experiment directory name after training

        # Run training
        try:
            trained_model_name = click_train(
                model_name,
                '40k',
                True,
                0,
                save_frequency,
                epochs,
                batch_size,
                True,
                'assets/pretrained_v2/f0G40k.pth',
                'assets/pretrained_v2/f0D40k.pth',
                gpu_count,
                cache,  # Use default cache value
                True,
                'v2',
            )
            print("Training completed successfully.")

            # Automatically save the trained model
            save_model_as_zip(trained_model_name)

        except Exception as e:
            print("An error occurred during training:", e)

#9. Training Calculator 

user_input = input("Calculator? (Y/N): ").strip().lower()

if user_input != 'y':
    print("Operation cancelled.")
else:
    import time

    # Input functions
    dataset_size = int(input("Enter the dataset size: "))
    batch_size = int(input("Specify batch size: "))
    epochs = int(input("Input number of epochs: "))
    epoch_per_second = float(input("Input epoch per second:"))

    # Calculate steps per epoch
    steps_per_epoch = dataset_size / batch_size

    # Calculate number of training cycles
    training_cycles = epochs / steps_per_epoch

    # Calculate cycle time
    cycle_time_seconds = epoch_per_second * epochs
    cycle_time_hours = int(cycle_time_seconds // 3600)
    cycle_time_minutes = int((cycle_time_seconds % 3600) // 60)
    cycle_time_seconds = int(cycle_time_seconds % 60)

    # Output results
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Number of training cycles: {training_cycles}")
    print(f"Cycle time: {cycle_time_hours} hours, {cycle_time_minutes} minutes, {cycle_time_seconds} seconds")

input("Press Enter to exit...")        
