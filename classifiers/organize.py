import os
import shutil
from tqdm.auto import tqdm
import argparse

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
    parser = argparse.ArgumentParser(description='Reorganize Dataset')
    parser.add_argument('--root_path', type=str, required=True, 
                        help='Path to the root directory of the dataset')
    parser.add_argument('--target_path',type=str, required=True,
                        help='Path to the tarjet directory for the new dataset')
    """                   
    if not os.path.isdir(args.root_path):
        raise ValueError(f"The provided root path does not exist: {args.root_path}")
    if not os.path.isdir(args.target_folder):
        raise ValueError(f"The provided target folder does not exist: {args.target_folder}")
    """
    args = parser.parse_args()

    reorganize_dataset(args.root_path, args.target_path)
