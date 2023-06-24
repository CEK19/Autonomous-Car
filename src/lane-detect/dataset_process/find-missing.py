import os
import shutil

def copy_missing_images(f1_folder, f2_folder, f3_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(f3_folder, exist_ok=True)
    
    # Create three subfolders in F3
    for i in range(3):
        subfolder = os.path.join(f3_folder, f"subfolder{i+1}")
        os.makedirs(subfolder, exist_ok=True)
    
    # Get a list of all the image files in F1
    f1_files = [f for f in os.listdir(f1_folder) if os.path.isfile(os.path.join(f1_folder, f))]
    
    # Calculate the number of files to put in each subfolder
    num_files = len(f1_files)
    files_per_subfolder = num_files // 3
    remainder = num_files % 3
    
    # Loop through the files in F1 and copy any missing files to F3
    subfolder_index = 0
    subfolder_file_count = 0
    for f1_file in f1_files:
        f2_file = os.path.join(f2_folder, f1_file)
        f3_file = os.path.join(f3_folder, f"subfolder{subfolder_index+1}", f1_file)
        if not os.path.exists(f2_file):
            shutil.copy(os.path.join(f1_folder, f1_file), f3_file)
            subfolder_file_count += 1
            if subfolder_file_count >= files_per_subfolder + (1 if remainder > 0 else 0):
                subfolder_index += 1
                subfolder_file_count = 0
                remainder -= 1

copy_missing_images("./allImage","./test-allImage","./missing")