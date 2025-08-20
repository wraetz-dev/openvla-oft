# This script combines multiple sets of TFDS formatted training data into a single TFDS dataset.
# This script takes as a command line argument the dataset names to combine.
# This script will look in the ~/bluecat-trainingdata directory for the datasets.

import json
import os
import shutil
import sys
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

def find_version_subfolder(dataset_path: Path) -> Optional[Path]:
    """
    Find version subfolder in a dataset directory.
    Returns the version subfolder path if found, None otherwise.
    """
    for item in dataset_path.iterdir():

        if item.is_dir() and re.match(r"^\d+\.\d+\.\d+$", item.name):
            return item
    return None

def examine_dataset(dataset_name):
    """
    Examine a dataset and understand how to combine it with other datasets.
    1. Read the dataset_info.json file to understand how many files there are, the total dataset size, and the number of episodes per file.
    2. ls the directory to understand the naming scheme of the files.
    3. Return a dictionary with the following keys:
        - filename: The name of a data file in the dataset
            - file_size: The number of episodes in the file
    """
    dataset_path = Path.home() / "bluecat-trainingdata" / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
    
    # Check if there's a version subfolder (like 0.0.1)
    version_subfolder = find_version_subfolder(dataset_path)
    
    # Determine the path to the actual data files
    if version_subfolder:
        data_path = version_subfolder
        info_file = version_subfolder / "dataset_info.json"
    else:
        data_path = dataset_path
        info_file = dataset_path / "dataset_info.json"
    
    # Read dataset_info.json
    with open(info_file, 'r') as f:
        dataset_info = json.load(f)
    
    # Get the train split info
    train_split = None
    for split in dataset_info['splits']:
        if split['name'] == 'train':
            train_split = split
            break
    
    if not train_split:
        raise ValueError(f"No train split found in dataset {dataset_name}")
    
    # Create mapping of filenames to episode counts
    file_info = {}
    shard_lengths = train_split['shardLengths']
    
    for i, episode_count in enumerate(shard_lengths):
        filename = f"{dataset_name}-train.tfrecord-{i:05d}"
        file_info[filename] = {
            'file_size': int(episode_count),
            'source_dataset': dataset_name
        }
    
    return {
        'dataset_info': dataset_info,
        'file_info': file_info,
        'total_episodes': sum(int(length) for length in shard_lengths),
        'total_bytes': int(train_split['numBytes'])
    }

def combine_datasets(dataset_names, combined_name=None):
    """
    Combine multiple datasets into a single TFDS dataset.
    1. Create a new folder in the ~/bluecat-trainingdata directory with the name of the combined dataset.
    2. For each input dataset, call examine_dataset to understand how to combine it with other datasets.
    3. Take the results of examine_dataset and create a new dataset_info.json file for the combined dataset.
    4. Copy the files into the new dataset directory. Change the file names to be consistent with the naming scheme.
    
    Args:
        dataset_names: List of dataset names to combine
        combined_name: Optional custom name for the combined dataset. If None, uses concatenated dataset names.
    """
    if len(dataset_names) < 2:
        raise ValueError("At least 2 datasets must be specified for combination")
    
    # Create combined dataset name
    if combined_name is None:
        combined_name = "_".join(dataset_names)
    base_path = Path.home() / "bluecat-trainingdata"
    
    # Get version from the first dataset (assuming all datasets have the same version)
    first_dataset_info = examine_dataset(dataset_names[0])
    version = first_dataset_info['dataset_info']['version']
    
    # Create version directory and combined dataset directory
    version_path = base_path / combined_name / version
    combined_path = version_path
    
    if combined_path.exists():
        raise FileExistsError(f"Combined dataset {combined_name} version {version} already exists")
    
    # Create the combined dataset directory
    combined_path.mkdir(parents=True, exist_ok=True)
    
    # Examine all datasets
    dataset_examinations = {}
    total_episodes = 0
    total_bytes = 0
    all_file_info = {}
    
    for dataset_name in dataset_names:
        print(f"Examining dataset: {dataset_name}")
        examination = examine_dataset(dataset_name)
        dataset_examinations[dataset_name] = examination
        total_episodes += examination['total_episodes']
        total_bytes += examination['total_bytes']
        
        # Add files to the combined file info with new naming
        for filename, file_data in examination['file_info'].items():
            # Create new filename for combined dataset
            new_filename = f"{combined_name}-train.tfrecord-{len(all_file_info):05d}"
            all_file_info[new_filename] = {
                'file_size': file_data['file_size'],
                'source_dataset': file_data['source_dataset'],
                'original_filename': filename
            }
    
    # Create combined dataset_info.json
    shard_lengths = [str(file_data['file_size']) for file_data in all_file_info.values()]
    
    combined_dataset_info = {
        "fileFormat": "tfrecord",
        "name": combined_name,
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}",
                "name": "train",
                "numBytes": str(total_bytes),
                "shardLengths": shard_lengths
            }
        ],
        "version": "0.0.1"
    }
    
    # Write combined dataset_info.json
    with open(combined_path / "dataset_info.json", 'w') as f:
        json.dump(combined_dataset_info, f, indent=2)
    
    # Copy features.json from the first dataset (assuming they're all compatible)
    first_dataset = dataset_names[0]
    first_dataset_path = base_path / first_dataset
    
    # Check if there's a version subfolder for the first dataset
    version_subfolder = find_version_subfolder(first_dataset_path)
    
    if version_subfolder:
        features_source = version_subfolder / "features.json"
    else:
        features_source = first_dataset_path / "features.json"
    
    if features_source.exists():
        shutil.copy2(features_source, combined_path / "features.json")
    
    # Copy all files with new names
    print(f"Copying files to combined dataset: {combined_name}")
    for new_filename, file_data in all_file_info.items():
        source_dataset = file_data['source_dataset']
        original_filename = file_data['original_filename']
        
        # Determine the source path, handling version subfolders
        source_dataset_path = base_path / source_dataset
        # Check if there's a version subfolder
        version_subfolder = find_version_subfolder(source_dataset_path)
        
        if version_subfolder:
            source_path = version_subfolder / original_filename
        else:
            source_path = source_dataset_path / original_filename
        
        dest_path = combined_path / new_filename
        
        print(f"  Copying {original_filename} -> {new_filename}")
        shutil.copy2(source_path, dest_path)
    
    print(f"\nCombined dataset created successfully!")
    print(f"Combined dataset name: {combined_name}")
    print(f"Version: {version}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total files: {len(all_file_info)}")
    print(f"Location: {combined_path}")

def main():
    """Main function to handle command line arguments and execute the combination."""
    if len(sys.argv) < 3:
        print("Usage: python combine_training_data.py [--name COMBINED_NAME] <dataset1> <dataset2> [dataset3] ...")
        print("Example: python combine_training_data.py evo_center dodge_center v2_tank_center")
        print("Example: python combine_training_data.py --name my_combined_dataset evo_center dodge_center v2_tank_center")
        sys.exit(1)
    
    # Parse command line arguments
    combined_name = None
    dataset_names = []
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ["--name", "-n"]:
            if i + 1 >= len(sys.argv):
                print("Error: --name flag requires a value")
                sys.exit(1)
            combined_name = sys.argv[i + 1]
            i += 2
        else:
            dataset_names.append(arg)
            i += 1
    
    if len(dataset_names) < 2:
        print("Error: At least 2 datasets must be specified")
        print("Usage: python combine_training_data.py [--name COMBINED_NAME] <dataset1> <dataset2> [dataset3] ...")
        sys.exit(1)
    
    try:
        combine_datasets(dataset_names, combined_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()