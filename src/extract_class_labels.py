#!/usr/bin/env python3
"""
Extract class labels from the processed GTSRB dataset for PYNQ deployment.
This script creates a separate file with class ID to sign name mapping.
"""

import os
import pickle
import json

# Hardcoded paths
DATASET_PATH = 'dataset/gtsrb_processed.pkl'
OUTPUT_PKL_PATH = 'models/class_labels.pkl'
OUTPUT_JSON_PATH = 'models/class_labels.json'


def extract_class_labels(dataset_path, output_pkl_path, output_json_path):
    """Extract class labels from the processed dataset and save to output files."""
    print(f"Loading dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    # Load the dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Extract class information
    num_classes = dataset.get('num_classes')
    class_names = dataset.get('class_names', {})
    
    if num_classes is None:
        raise ValueError("Dataset does not contain 'num_classes' field")
    
    print(f"Found {num_classes} classes in the dataset")
    
    # If class_names is not available, create generic mapping
    if not class_names:
        print("Warning: 'class_names' not found in dataset. Creating generic class names.")
        class_names = {i: f"Class {i}" for i in range(num_classes)}
    
    # Create class labels dictionary
    class_labels = {
        'num_classes': num_classes,
        'class_names': class_names
    }
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Save as pickle file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(class_labels, f)
    print(f"Class labels saved to pickle file: {output_pkl_path}")
    
    # Save as JSON file (more portable for different environments)
    with open(output_json_path, 'w') as f:
        # Convert integer keys to strings for JSON
        json_class_names = {str(k): v for k, v in class_names.items()}
        json_class_labels = {
            'num_classes': num_classes,
            'class_names': json_class_names
        }
        json.dump(json_class_labels, f, indent=2)
    print(f"Class labels saved to JSON file: {output_json_path}")
    
    # Print class mapping
    print("\nClass ID to name mapping:")
    for class_id in sorted(class_names.keys()):
        print(f"  {class_id}: {class_names[class_id]}")


def main():
    try:
        # Extract and save class labels
        extract_class_labels(DATASET_PATH, OUTPUT_PKL_PATH, OUTPUT_JSON_PATH)
        print("\nClass label extraction completed successfully")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
