import os

def delete_unmatched_files(dir_path_A, dir_path_B):
    # Get list of files in directory A
    files_A = os.listdir(dir_path_A)
    
    # Get list of files in directory B
    files_B = os.listdir(dir_path_B)
    
    # Check each file in directory A
    for file_A in files_A:
        # Check if file exists in directory B
        if file_A not in files_B:
            # If file doesn't exist in directory B, delete from directory A
            os.remove(os.path.join(dir_path_A, file_A))
            
            # Print message to confirm deletion
            print(f"{file_A} deleted from {dir_path_A}")

delete_unmatched_files("D:/container/AI_DCLV/readData/labeled_v6/image","D:/container/AI_DCLV/readData/labeled_v6/label")