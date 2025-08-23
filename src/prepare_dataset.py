#!/usr/bin/env python3
"""
Enhanced dataset preparation script for the GTSRB Road Sign dataset.
Creates a pickle file with preprocessed images ready for training.

Dataset structure:
- dataset/
    - Meta/
    - Train/
        - (0-42 folders with images for each class)
    - Test/
    - Meta.csv
    - Train.csv
    - Test.csv
"""

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

# Hard-coded paths
DATASET_PATH = 'dataset'
OUTPUT_PATH = 'dataset/gtsrb_processed.pkl'
SAMPLE_VIDEO_PATH = 'dataset/sample_road_signs.mp4'
IMG_SIZE = 32  # Square images (IMG_SIZE x IMG_SIZE)


def load_class_names(meta_csv_path):
    """Loads class ID to sign name mapping from Meta.csv."""
    if not os.path.exists(meta_csv_path):
        print(f"Warning: Meta.csv not found at {meta_csv_path}. Class names will be generic.")
        return None
    try:
        meta_df = pd.read_csv(meta_csv_path)
        
        # Check for necessary columns
        if 'ClassId' in meta_df.columns and 'SignName' in meta_df.columns:
            class_names = pd.Series(meta_df.SignName.values, index=meta_df.ClassId).to_dict()
        elif 'ClassId' in meta_df.columns and 'SignId' in meta_df.columns:
            # Alternative field names
            class_names = pd.Series(meta_df.SignId.values, index=meta_df.ClassId).to_dict()
        else:
            print(f"Warning: Meta.csv at {meta_csv_path} is missing required columns. Class names will be generic.")
            return None
            
        print(f"Loaded {len(class_names)} class names from {meta_csv_path}")
        return class_names
    except Exception as e:
        print(f"Error loading class names from {meta_csv_path}: {e}. Class names will be generic.")
        return None


def preprocess_image(img_path, roi=None):
    """Load, crop, and preprocess a single image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop to ROI if provided
    if roi is not None:
        x1, y1, x2, y2 = roi
        if (x1 < x2 and y1 < y2 and 
            x1 >= 0 and y1 >= 0 and 
            x2 <= img.shape[1] and y2 <= img.shape[0]):
            img = img[y1:y2, x1:x2]
    
    # Resize
    try:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"  > Error resizing image {img_path} (shape: {img.shape}): {e}")
        return None
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def process_dataset(dataset_path=DATASET_PATH, max_samples_per_class=None):
    """
    Process the GTSRB dataset using the provided CSV annotation files:
    - Reads images using paths from Train.csv and Test.csv
    - Extracts class information from ClassId column
    - Crops using bounding boxes from CSV
    - Loads class names from Meta.csv
    """
    print(f"Processing GTSRB dataset from: {dataset_path}")

    # --- Load Class Names ---
    meta_csv_path = os.path.join(dataset_path, 'Meta.csv')
    class_id_to_name_map = load_class_names(meta_csv_path)

    # --- Process Training Data using Train.csv ---
    train_images = []
    train_labels = []
    train_csv_path = os.path.join(dataset_path, 'Train.csv')

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV file not found at: {train_csv_path}")

    try:
        train_df = pd.read_csv(train_csv_path)
        
        if max_samples_per_class:
            print(f"Limiting samples per class to {max_samples_per_class}")
            train_df = train_df.groupby('ClassId').head(max_samples_per_class)
            
        total_train_images = len(train_df)
        print(f"Processing {total_train_images} training images from {train_csv_path}")

        for index, row in tqdm(train_df.iterrows(), total=total_train_images, desc="Training images"):
            img_relative_path = row['Path'].replace('\\', '/')
            img_path = os.path.join(dataset_path, img_relative_path)
            
            roi = [int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])]
            img = preprocess_image(img_path, roi)
            
            if img is not None:
                train_images.append(img)
                train_labels.append(int(row['ClassId']))

    except Exception as e:
        print(f"Error processing training data from {train_csv_path}: {e}")
        raise

    # --- Process Test Data using Test.csv ---
    test_images = []
    test_labels = []
    test_csv_path = os.path.join(dataset_path, 'Test.csv')

    if not os.path.exists(test_csv_path):
        print(f"Warning: Test CSV file not found at {test_csv_path}. Test set will be created from training data split.")
    else:
        try:
            test_df = pd.read_csv(test_csv_path)
            total_test_images = len(test_df)
            print(f"Processing {total_test_images} test images from {test_csv_path}")

            for index, row in tqdm(test_df.iterrows(), total=total_test_images, desc="Test images"):
                img_relative_path = row['Path'].replace('\\', '/')
                img_path = os.path.join(dataset_path, img_relative_path)
                
                roi = [int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])]
                img = preprocess_image(img_path, roi)
                
                if img is not None:
                    test_images.append(img)
                    test_labels.append(int(row['ClassId']))

        except Exception as e:
            print(f"Error processing test data from {test_csv_path}: {e}")
            print("Will split training data to create test set instead.")
            test_images, test_labels = [], []

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    if len(train_images) == 0:
        raise ValueError("No training data was successfully processed!")

    if len(test_images) == 0:
        print("Creating train/val/test split from training data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            train_images, train_labels, test_size=0.3, random_state=42, stratify=train_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp  # 15% val, 15% test
        )
    else:
        print("Creating train/val split from training data (using loaded test data)...")
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        X_test = np.array(test_images)
        y_test = np.array(test_labels)

    # Determine num_classes from all labels
    all_labels = np.concatenate([y_train, y_val, y_test])
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} unique classes in the data.")

    # If class_id_to_name_map is None or incomplete, create a generic one
    if class_id_to_name_map is None:
        class_id_to_name_map = {i: f"Class {i}" for i in unique_labels}
        print("Using generic class names as Meta.csv was not found or was invalid.")
    else:
        # Ensure all unique_labels have a name
        for label_id in unique_labels:
            if label_id not in class_id_to_name_map:
                print(f"Warning: ClassId {label_id} found in data but not in Meta.csv. Assigning generic name.")
                class_id_to_name_map[label_id] = f"Class {label_id}"

    dataset = {
        'train_data': X_train,
        'train_labels': y_train,
        'val_data': X_val,
        'val_labels': y_val,
        'test_data': X_test,
        'test_labels': y_test,
        'num_classes': num_classes,
        'class_names': class_id_to_name_map
    }

    print(f"Final dataset splits:")
    print(f"  Training:   {len(X_train)} images, {len(np.unique(y_train))} classes")
    print(f"  Validation: {len(X_val)} images, {len(np.unique(y_val))} classes")
    print(f"  Test:       {len(X_test)} images, {len(np.unique(y_test))} classes")

    return dataset


def save_dataset(dataset, output_path=OUTPUT_PATH):
    """Save the processed dataset to a pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Processed dataset saved to {output_path}")


def save_sample_video(dataset, output_path=SAMPLE_VIDEO_PATH, num_frames=100):
    """Create a sample video from test images for visualization."""
    try:
        test_data = dataset['test_data']
        test_labels = dataset['test_labels']
        class_names = dataset.get('class_names', {})
        
        if len(test_data) == 0:
            print("No test data available to create sample video.")
            return

        if len(test_data) > num_frames:
            indices = np.random.choice(len(test_data), num_frames, replace=False)
            test_data = test_data[indices]
            test_labels = test_labels[indices] if len(test_labels) > 0 else []

        target_size = (320, 240)
        frames = []
        print(f"Creating sample video with {len(test_data)} frames...")
        for idx, (img, label) in enumerate(zip(test_data, test_labels)):
            img_uint8 = (img * 255).astype(np.uint8)
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
            # Resize image for display
            h, w = img_uint8.shape[:2]
            if h == 0 or w == 0: continue
            new_h, new_w = (200, int(w * (200 / h))) if h > w else (int(h * (200 / w)), 200)
            if new_w <= 0 or new_h <= 0: continue
            
            try:
                resized = cv2.resize(img_uint8, (new_w, new_h))
            except cv2.error:
                continue
                
            # Place image on canvas
            x_offset, y_offset = (target_size[0] - new_w) // 2, 20
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Add text label
            class_name = class_names.get(label, f"Class {label}")
            label_text = f"Class: {label} - {class_name}"
            
            # Add text with background
            cv2.rectangle(canvas, (10, target_size[1]-40), (target_size[0]-10, target_size[1]-10), (0, 0, 0), -1)
            cv2.putText(canvas, label_text, (20, target_size[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            frames.append(canvas)

        if not frames: 
            print("No valid frames generated for sample video.")
            return
            
        # Create video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, 3, target_size)  # 3 fps for better viewing
        for frame in frames: 
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()
        print(f"Sample video created at {output_path}")
    except Exception as e:
        print(f"Error creating sample video: {e}")


if __name__ == '__main__':
    try:
        # Process the dataset
        processed_dataset = process_dataset(DATASET_PATH)
        
        # Save the full dataset
        save_dataset(processed_dataset, OUTPUT_PATH)
        
        # Create a subset for testing (optional)
        if 'test_data' in processed_dataset and len(processed_dataset['test_data']) > 0:
            subset_size = min(100, len(processed_dataset['test_data']))
            test_subset = {
                'test_data': processed_dataset['test_data'][:subset_size],
                'test_labels': processed_dataset['test_labels'][:subset_size],
                'num_classes': processed_dataset['num_classes'],
                'class_names': processed_dataset['class_names'] 
            }
            save_dataset(test_subset, 'dataset/gtsrb_test_subset.pkl')
            print(f"Saved test subset with {subset_size} samples")
        
        # Create a sample video
        save_sample_video(processed_dataset, SAMPLE_VIDEO_PATH)
        
        print("\nDataset preparation complete!")
        
    except Exception as e:
        print(f"\nAn error occurred during dataset preparation: {e}")
        import traceback
        traceback.print_exc()