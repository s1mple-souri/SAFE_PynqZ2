#!/usr/bin/env python3
"""
Video-based traffic sign recognition script.
MODIFIED: 
- Uses the 2-block QuantRoadSignNet from 'train_modified.py'.
- Uses relative paths for input video, model, labels, and output.
- Imports model definition from 'train_modified.py'.
"""

import cv2
import torch
import pickle
import numpy as np
import os
import sys

# --- Path Configuration and Model Import ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) # Assumes src is child of Project

# Add src to sys.path to allow direct import of train_modified
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Add project root to sys.path if needed for other modules (though not strictly for train_modified here)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Import the modified 2-block model
    from train import QuantRoadSignNet
    print("Successfully imported QuantRoadSignNet from train_modified.py")
except ImportError as e:
    print(f"Error importing QuantRoadSignNet from train_modified.py: {e}")
    print("Ensure train_modified.py (defining the 2-block QuantRoadSignNet) is in the same directory or Python path.")
    print(f"Current SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"Current PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Relative Path Configuration ---
# These paths are relative to the PROJECT_ROOT directory (e.g., 'Project/')
DEFAULT_VIDEO_INPUT_PATH = os.path.join(PROJECT_ROOT, "dataset", "sample_road_signs.mp4")
# Path to the modified (2-block) model weights
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rsc_w4a4_w4a4_finn.pth")
DEFAULT_CLASS_LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "class_labels.pkl")
DEFAULT_VIDEO_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "processed_video_2block_new.mp4")

DEFAULT_NUM_CLASSES = 43  # GTSRB has 43 classes
TARGET_SIZE = (32, 32) # Input size for the model
DEFAULT_CONF_THRESHOLD = 0.5

def load_pytorch_model(model_path, num_classes, device):
    """Load the trained PyTorch (Brevitas) model."""
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model has been trained and saved correctly using 'train_modified.py'.")
    
    model = QuantRoadSignNet(num_classes=num_classes) # Uses the 2-block model definition
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def load_class_labels(labels_path):
    """Load class labels (ID to name mapping)."""
    print(f"Loading class labels from: {labels_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Class labels file not found at {labels_path}. Ensure 'export_to_onnx_modified.py' or 'prepare_dataset_modified.py' generated this.")
    
    with open(labels_path, "rb") as f:
        class_info = pickle.load(f)
    print("Class labels loaded successfully.")
    # Ensure class_names is a dictionary, provide default if key missing
    class_names_map = class_info.get("class_names", {i: f"Class {i}" for i in range(class_info.get("num_classes", DEFAULT_NUM_CLASSES))})
    return class_names_map, class_info.get("num_classes", DEFAULT_NUM_CLASSES)

def preprocess_frame(frame_bgr, target_size=(32, 32)):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(normalized_frame).permute(2, 0, 1)
    frame_tensor = frame_tensor.unsqueeze(0)
    return frame_tensor

def predict_on_frame(model, frame_tensor, device):
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        outputs = model(frame_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class_id = torch.max(probabilities, 1)
    return predicted_class_id.item(), confidence.item()

def main(video_path, output_video_path, model_path, labels_path, conf_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        class_names, num_classes_from_labels = load_class_labels(labels_path)
        # Use num_classes from labels file if available, otherwise default
        num_classes_to_use = num_classes_from_labels if num_classes_from_labels is not None else DEFAULT_NUM_CLASSES
        model = load_pytorch_model(model_path, num_classes_to_use, device)

    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Processing video: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: Video input file not found at {video_path}. It should be at {DEFAULT_VIDEO_INPUT_PATH} or specified.")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # Default FPS if not readable

    out_writer = None
    if output_video_path:
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e_dir:
                print(f"Error creating output directory {output_dir}: {e_dir}")
                output_video_path = None 

        if output_video_path:
            print(f"Output video will be saved to: {output_video_path}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        input_tensor = preprocess_frame(frame, target_size=TARGET_SIZE)
        predicted_id, confidence = predict_on_frame(model, input_tensor, device)
        
        label_text = "Unknown"
        if confidence >= conf_threshold:
            label_text = class_names.get(predicted_id, f"ClassID {predicted_id}") # Use ClassID if name missing
            label_text = f"{label_text} ({confidence:.2f})"
        else:
            label_text = f"Low Conf ({confidence:.2f})"

        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display frame (optional, can be slow)
        # cv2.imshow("Video Inference", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        if out_writer:
            out_writer.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    if out_writer:
        out_writer.release()
    # cv2.destroyAllWindows() # If using imshow

    print(f"Finished processing video. Total frames: {frame_count}")
    if output_video_path and os.path.exists(output_video_path):
        print(f"Output video saved to {output_video_path}")
    elif output_video_path:
        print(f"Output video was intended for {output_video_path} but may not have been saved.")

if __name__ == "__main__":
    print("Running video inference with relative paths for the 2-block model.")
    # Create dummy dataset and model files if they don't exist for basic script run without full pipeline execution
    # This is for basic testing of the script structure, not functional correctness of the model.
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"Warning: Model file {DEFAULT_MODEL_PATH} not found. Video inference will likely fail to load the model.")
        print("Please run 'train_modified.py' first to generate the model file.")
    if not os.path.exists(DEFAULT_CLASS_LABELS_PATH):
        print(f"Warning: Class labels file {DEFAULT_CLASS_LABELS_PATH} not found. Video inference may use default class names.")
        print("Please run 'export_to_onnx_modified.py' or 'prepare_dataset_modified.py' to generate class labels.")
    if not os.path.exists(DEFAULT_VIDEO_INPUT_PATH):
        print(f"Warning: Sample video {DEFAULT_VIDEO_INPUT_PATH} not found. Attempting to create a dummy video.")
        # Create a dummy video if none exists
        os.makedirs(os.path.dirname(DEFAULT_VIDEO_INPUT_PATH), exist_ok=True)
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Dummy Video", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        dummy_writer = cv2.VideoWriter(DEFAULT_VIDEO_INPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 1, (320,240))
        for _ in range(10):
            dummy_writer.write(dummy_frame)
        dummy_writer.release()
        print(f"Created dummy video at {DEFAULT_VIDEO_INPUT_PATH}")

    main(video_path=DEFAULT_VIDEO_INPUT_PATH,
         output_video_path=DEFAULT_VIDEO_OUTPUT_PATH,
         model_path=DEFAULT_MODEL_PATH,
         labels_path=DEFAULT_CLASS_LABELS_PATH,
         conf_threshold=DEFAULT_CONF_THRESHOLD)

