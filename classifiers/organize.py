import os
import shutil
from tqdm.auto import tqdm

def reorganize_dataset(root_path, target_folder):
    categories = ['bcc', 'mel', 'scc']
    for cat in categories:
        os.makedirs(os.path.join(target_folder, cat), exist_ok=True)

    # Count total number of files for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(root_path))

    progress_bar = tqdm(total=total_files, desc="Processing Files", unit="file")
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if any(cat in file for cat in categories):
                source_file = os.path.join(dirpath, file)
                for cat in categories:
                    if cat in file:
                        dest_file = os.path.join(target_folder, cat, file)
                        shutil.copy2(source_file, dest_file)
                        break  # Break the loop once the file is copied to a category
            progress_bar.update(1)
    progress_bar.close()

if __name__ == "__main__":
    root_path = '../data/val/others/'
    target_folder = '../data_mult/val/'
    reorganize_dataset(root_path, target_folder)
