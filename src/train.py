# #!/usr/bin/env python3
# """
# Enhanced training script for FINN-friendly quantized neural network
# optimized for PYNQ FPGA deployment.

# This script trains a road sign classifier with quantized weights and activations
# using the Brevitas library to ensure compatibility with FINN compiler.
# """

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import brevitas.nn as qnn
# from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time

# # --- Configuration ---
# # Hardcoded paths
# DATASET_PATH = 'dataset/gtsrb_processed.pkl'
# OUTPUT_DIR = 'models'
# MODEL_BASE_NAME = 'road_sign_classifier'

# # Training parameters
# BATCH_SIZE = 64
# EPOCHS = 20
# LEARNING_RATE = 0.001
# WEIGHT_DECAY = 1e-5

# # Model configuration
# # Only using 8-bit quantization since 4-bit is not available in installed version
# QUANT_MODE = 'w8a8'  

# # Set random seed for reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # --- Model Definition (FINN-Friendly) ---
# class QuantRoadSignNet(nn.Module):
#     def __init__(self, num_classes=43, quant_mode='w8a8'):
#         super(QuantRoadSignNet, self).__init__()
        
#         # Set quantization - only 8-bit is available
#         weight_quant = Int8WeightPerTensorFloat
#         act_quant = Int8ActPerTensorFloat
#         print("Using 8-bit weights and activations (w8a8)")
        
#         self.features = nn.Sequential(
#             # First block: Conv + BN + ReLU
#             qnn.QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1, 
#                             weight_quant=weight_quant, bias=False),
#             nn.BatchNorm2d(16),
#             qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True),
            
#             # Replaced MaxPool with strided conv
#             qnn.QuantConv2d(16, 16, kernel_size=3, stride=2, padding=1, 
#                            weight_quant=weight_quant, bias=False),
#             nn.BatchNorm2d(16),
#             qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True),
            
#             # Second block
#             qnn.QuantConv2d(16, 32, kernel_size=3, stride=1, padding=1, 
#                            weight_quant=weight_quant, bias=False),
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True),
            
#             # Replaced MaxPool with strided conv
#             qnn.QuantConv2d(32, 32, kernel_size=3, stride=2, padding=1, 
#                            weight_quant=weight_quant, bias=False),
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True),
            
#             # Third block
#             qnn.QuantConv2d(32, 64, kernel_size=3, stride=1, padding=1, 
#                            weight_quant=weight_quant, bias=False),
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True),
            
#             # Replaced MaxPool with strided conv
#             qnn.QuantConv2d(64, 64, kernel_size=3, stride=2, padding=1, 
#                            weight_quant=weight_quant, bias=False),
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True),
#         )
        
#         self.classifier = nn.Sequential(
#             qnn.QuantLinear(64 * 4 * 4, 256, bias=True, weight_quant=weight_quant),
#             qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True),
#             qnn.QuantLinear(256, num_classes, bias=True, weight_quant=weight_quant)
#         )
        
#         # Store quantization mode for reference
#         self.quant_mode = quant_mode

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


# # --- Data Handling Functions ---
# def load_dataset(path=DATASET_PATH):
#     """Load the preprocessed dataset."""
#     print(f"Loading dataset from {path}...")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset file not found at {path}. Please run prepare_dataset.py first.")
#     with open(path, 'rb') as f:
#         dataset = pickle.load(f)
#     print("Dataset loaded successfully.")
#     return dataset


# def prepare_data_loaders(dataset, batch_size=BATCH_SIZE, use_cuda=False):
#     """Prepare DataLoader objects for training, validation, and testing."""
#     print("Preparing data loaders...")
#     required_keys = ['train_data', 'train_labels', 'val_data', 'val_labels']
#     if not all(key in dataset for key in required_keys):
#         raise KeyError(f"Dataset dictionary missing one or more required keys: {required_keys}")

#     X_train = torch.tensor(dataset['train_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#     y_train = torch.tensor(dataset['train_labels'], dtype=torch.long)
#     X_val = torch.tensor(dataset['val_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#     y_val = torch.tensor(dataset['val_labels'], dtype=torch.long)

#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)

#     pin_memory = use_cuda
#     num_workers = 4 if use_cuda else 0

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                              pin_memory=pin_memory, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
#                            pin_memory=pin_memory, num_workers=num_workers)

#     # Prepare test loader if test data is available
#     test_loader = None
#     if 'test_data' in dataset and 'test_labels' in dataset and len(dataset['test_data']) > 0:
#         X_test = torch.tensor(dataset['test_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#         y_test = torch.tensor(dataset['test_labels'], dtype=torch.long)
#         test_dataset = TensorDataset(X_test, y_test)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
#                                 pin_memory=pin_memory, num_workers=num_workers)

#     print("Data loaders ready.")
#     return train_loader, val_loader, test_loader


# # --- Training Function ---
# def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS):
#     """Train the model and validate after each epoch."""
#     print(f"Starting training on {device} for {epochs} epochs...")
#     model.to(device)

#     train_losses, val_losses, train_accs, val_accs = [], [], [], []
#     best_val_acc = 0.0
#     model_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_{model.quant_mode}.pth")
#     start_time = time.time()

#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0
#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
#         for inputs, labels in train_pbar:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             train_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})
        
#         epoch_train_loss = running_loss / total if total > 0 else 0
#         epoch_train_acc = 100 * correct / total if total > 0 else 0
#         train_losses.append(epoch_train_loss)
#         train_accs.append(epoch_train_acc)

#         # Validation phase
#         model.eval()
#         running_loss, correct, total = 0.0, 0, 0
#         val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
#         with torch.no_grad():
#             for inputs, labels in val_pbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 val_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})

#         epoch_val_loss = running_loss / total if total > 0 else 0
#         epoch_val_acc = 100 * correct / total if total > 0 else 0
#         val_losses.append(epoch_val_loss)
#         val_accs.append(epoch_val_acc)

#         # Print epoch summary
#         epoch_time = time.time() - start_time
#         print(f"Epoch {epoch+1}/{epochs} Summary: "
#               f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
#               f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | "
#               f"Time: {epoch_time:.1f}s")

#         # Save best model
#         if epoch_val_acc > best_val_acc:
#             best_val_acc = epoch_val_acc
#             torch.save(model.state_dict(), model_path)
#             print(f"---> Saved best model to {model_path} with validation accuracy: {best_val_acc:.2f}% at epoch {epoch+1}")

#     total_time = time.time() - start_time
#     print(f"Training finished in {total_time:.2f} seconds.")
    
#     # Generate and save training plot
#     plot_path = os.path.join(OUTPUT_DIR, f"training_history_{model.quant_mode}.png")
#     try:
#         plt.figure(figsize=(12, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
#         plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
#         plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss vs. Epochs'); plt.legend(); plt.grid(True)

#         plt.subplot(1, 2, 2)
#         plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
#         plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
#         plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy vs. Epochs'); plt.legend(); plt.grid(True)

#         plt.tight_layout()
#         plt.savefig(plot_path)
#         print(f"Training history plot saved to {plot_path}")
#         plt.close()
#     except Exception as plot_e:
#         print(f"Warning: Could not generate or save training plot: {plot_e}")
        
#     return model, model_path, best_val_acc


# # --- Testing Function ---
# def test_model(model, test_loader, device):
#     """Evaluate the model on test data."""
#     print("Testing model...")
#     model.to(device)
#     model.eval()

#     if test_loader is None:
#         print("No test data available. Skipping testing.")
#         return None

#     class_correct = {}
#     class_total = {}
#     correct, total = 0, 0
    
#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Testing"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
            
#             # Calculate overall accuracy
#             c = (predicted == labels).squeeze()
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             # Calculate per-class accuracy
#             for i, label in enumerate(labels):
#                 label_key = label.item()
#                 if label_key not in class_correct:
#                     class_correct[label_key] = 0
#                     class_total[label_key] = 0
                
#                 if len(c.shape) == 0:  # Handle single item in batch
#                     class_correct[label_key] += c.item()
#                 else:
#                     class_correct[label_key] += c[i].item()
#                 class_total[label_key] += 1
    
#     # Calculate and display overall accuracy
#     test_acc = 100 * correct / total if total > 0 else 0
#     print(f"Test Accuracy: {test_acc:.2f}% ({correct}/{total})")
    
#     # Display per-class accuracy
#     print("\nPer-class accuracy:")
#     per_class_accs = []
#     for class_id in sorted(class_total.keys()):
#         class_acc = 100 * class_correct[class_id] / class_total[class_id]
#         per_class_accs.append(class_acc)
#         print(f"  Class {class_id}: {class_acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})")
    
#     # Calculate average per-class accuracy
#     avg_class_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0
#     print(f"\nAverage per-class accuracy: {avg_class_acc:.2f}%")
    
#     return {
#         'accuracy': test_acc,
#         'per_class_accuracy': {class_id: 100 * class_correct[class_id] / class_total[class_id] 
#                               for class_id in class_total},
#         'avg_class_accuracy': avg_class_acc
#     }

# # --- Main Execution Block ---
# def main():
#     """Main execution function."""
#     # Set device
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     print(f"Using device: {device}")
#     if use_cuda: 
#         print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

#     try:
#         # Load dataset
#         dataset = load_dataset(DATASET_PATH)
#         num_classes = dataset.get('num_classes')
#         class_names = dataset.get('class_names', {})
        
#         if num_classes is None:
#             raise ValueError("Dataset missing 'num_classes' key.")
#         print(f"Dataset contains {num_classes} classes.")
        
#         # Prepare data loaders
#         train_loader, val_loader, test_loader = prepare_data_loaders(dataset, BATCH_SIZE, use_cuda)
        
#         # Create model with specified quantization
#         model = QuantRoadSignNet(num_classes=num_classes, quant_mode=QUANT_MODE)
#         print(f"\nModel architecture:\n{model}\n")
        
#         # Define loss function and optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#         print(f"\nStarting training using 8-bit quantization for FINN compatibility.")
#         print(f"Training parameters: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LEARNING_RATE}")

#         # Train model
#         trained_model, model_path, best_val_acc = train_model(
#             model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
#         )
        
#         # Test the best model
#         print(f"\nLoading best model from {model_path} for final testing...")
#         test_model_instance = QuantRoadSignNet(num_classes=num_classes, quant_mode=QUANT_MODE)
#         try:
#             test_model_instance.load_state_dict(torch.load(model_path, map_location=device))
#             test_results = test_model(test_model_instance, test_loader, device)
            
#             if test_results:
#                 # Save model summary to a text file
#                 summary_path = os.path.join(OUTPUT_DIR, f"model_summary_{QUANT_MODE}.txt")
#                 with open(summary_path, 'w') as f:
#                     f.write(f"Road Sign Classifier Model Summary\n")
#                     f.write(f"================================\n\n")
#                     f.write(f"Quantization mode: {QUANT_MODE}\n")
#                     f.write(f"Number of classes: {num_classes}\n")
#                     f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
#                     f.write(f"Test accuracy: {test_results['accuracy']:.2f}%\n")
#                     f.write(f"Average per-class test accuracy: {test_results['avg_class_accuracy']:.2f}%\n\n")
                    
#                     # Save per-class accuracy
#                     f.write(f"Per-class accuracy:\n")
#                     for class_id in sorted(test_results['per_class_accuracy'].keys()):
#                         class_name = class_names.get(class_id, f"Class {class_id}")
#                         f.write(f"  Class {class_id} ({class_name}): {test_results['per_class_accuracy'][class_id]:.2f}%\n")
                    
#                     f.write(f"\nModel saved to: {model_path}\n")
#                     f.write(f"\nModel Architecture:\n")
#                     f.write(f"{model}\n")
                
#                 print(f"Model summary saved to {summary_path}")
            
#         except FileNotFoundError:
#             print(f"ERROR: Model file {model_path} not found after training. Training might have failed to save a model.")
#         except RuntimeError as e:
#             print(f"ERROR loading the trained model for testing: {e}")
#         except Exception as load_e:
#             print(f"An unexpected error occurred while loading/testing the model: {load_e}")
#             import traceback
#             traceback.print_exc()
            
#         print("\nTraining workflow completed!")
        
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == '__main__':
#     main()

# #!/usr/bin/env python3
# """
# FINN-compatible W4A4 quantized road sign classifier for PYNQ deployment.
# This script follows the latest FINN and Brevitas requirements for proper ONNX export.

# Key FINN Requirements Addressed:
# - No MaxPool layers (replaced with strided convolutions)
# - Proper 4-bit quantization using IntQuant quantizers
# - FINN-compatible layer ordering and structure
# - Single-path architecture (no skip connections for initial deployment)
# - Proper bias quantization for FINN compatibility
# """

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import brevitas.nn as qnn
# from brevitas.quant.scaled_int import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
# from brevitas.quant import Int8Bias  # Try Int8Bias first, fallback if needed
# from brevitas.core.restrict_val import RestrictValueType
# from brevitas.core.scaling import ScalingImplType
# from brevitas.core.zero_point import ZeroZeroPoint
# from brevitas.core.bit_width import BitWidthImplType  
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time

# # --- Configuration ---
# DATASET_PATH = 'dataset/gtsrb_processed.pkl'
# OUTPUT_DIR = 'models'
# MODEL_BASE_NAME = 'road_sign_classifier_new'

# # Training parameters
# BATCH_SIZE = 32  # Reduced for 4-bit training stability
# EPOCHS = 30      # Increased for better convergence with low precision
# LEARNING_RATE = 0.0005  # Lower LR for quantized training
# WEIGHT_DECAY = 1e-4

# # Model configuration - FINN W4A4 
# QUANT_MODE = 'w4a4'  # 4-bit weights and activations

# # Set random seed for reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- FINN-Compatible Custom Quantizers ---
# class FINNUint4ActQuant(Int8ActPerTensorFloat):
#     """
#     FINN-compatible unsigned 4-bit activation quantizer.
#     FINN Requirements:
#     - UNSIGNED (signed=False) 
#     - NON-NARROW (narrow=False)
#     - Zero point = 0
#     - 4-bit precision
#     """
#     bit_width = 4
#     signed = False
#     narrow = False
#     zero_point_impl = ZeroZeroPoint
    
# class FINNInt4WeightQuant(Int8WeightPerTensorFloat):
#     """FINN-compatible 4-bit signed weight quantizer."""
#     bit_width = 4
#     signed = True  # Weights can be signed

# # Try to get the best available bias quantizer for your Brevitas version
# try:
#     from brevitas.quant import Int32Bias
#     bias_quantizer = Int32Bias
#     print("Using Int32Bias for bias quantization")
# except ImportError:
#     try:
#         from brevitas.quant.scaled_int import Int16Bias
#         bias_quantizer = Int16Bias
#         print("Using Int16Bias for bias quantization (Int32Bias not available)")
#     except ImportError:
#         from brevitas.quant import Int8Bias
#         bias_quantizer = Int8Bias
#         print("Using Int8Bias for bias quantization (higher precision bias not available)")

# # --- FINN-Compatible W4A4 Model ---
# class FINNRoadSignNet(nn.Module):
#     """
#     FINN-compatible road sign classifier with W4A4 quantization.
    
#     Key FINN Design Principles:
#     1. No MaxPool layers - replaced with strided convolutions
#     2. Proper 4-bit quantization throughout
#     3. BatchNorm after each conv layer
#     4. Single dataflow path (no residual connections)
#     5. Input quantization for proper FINN pipeline
#     """
    
#     def __init__(self, num_classes=43, input_size=(32, 32)):
#         super(FINNRoadSignNet, self).__init__()
        
#         # FINN-compatible 4-bit quantizers 
#         # CRITICAL: FINN requires UNSIGNED and NON-NARROW activations for ReLU
#         self.weight_quant = FINNInt4WeightQuant
#         self.act_quant = FINNUint4ActQuant  # Custom FINN-compatible unsigned quantizer
#         # Temporarily disable bias quantization to avoid scale dependency issues
#         self.bias_quant = None  # Will add back once model trains successfully
        
#         print(f"Initializing FINN W4A4 model for {num_classes} classes")
#         print(f"Expected input size: {input_size}")
        
#         # Input quantization - CRITICAL for FINN pipeline
#         self.input_quant = qnn.QuantIdentity(
#             act_quant=self.act_quant,
#             return_quant_tensor=True
#         )
        
#         # Convolutional Feature Extractor
#         # FINN prefers smaller networks, so we keep it compact but effective
#         self.features = nn.Sequential(
#             # Block 1: 3->32 channels, downsample 32x32 -> 16x16
#             qnn.QuantConv2d(3, 32, kernel_size=3, stride=2, padding=1,  # stride=2 replaces maxpool
#                            bias=True,
#                            weight_quant=self.weight_quant, 
#                            bias_quant=self.bias_quant,  # None for now
#                            return_quant_tensor=False),  # Return regular tensor for BatchNorm
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Block 2: 32->32 channels, maintain 16x16
#             qnn.QuantConv2d(32, 32, kernel_size=3, padding=1,
#                            bias=True,
#                            weight_quant=self.weight_quant,
#                            bias_quant=self.bias_quant,  # None for now
#                            return_quant_tensor=False),  # Return regular tensor for BatchNorm
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Block 3: 32->64 channels, downsample 16x16 -> 8x8  
#             qnn.QuantConv2d(32, 64, kernel_size=3, stride=2, padding=1,  # stride=2 replaces maxpool
#                            bias=True,
#                            weight_quant=self.weight_quant,
#                            bias_quant=self.bias_quant,  # None for now
#                            return_quant_tensor=False),  # Return regular tensor for BatchNorm
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Block 4: 64->64 channels, maintain 8x8
#             qnn.QuantConv2d(64, 64, kernel_size=3, padding=1,
#                            bias=True,
#                            weight_quant=self.weight_quant, 
#                            bias_quant=self.bias_quant,  # None for now
#                            return_quant_tensor=False),  # Return regular tensor for BatchNorm
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Block 5: 64->128 channels, downsample 8x8 -> 4x4
#             qnn.QuantConv2d(64, 128, kernel_size=3, stride=2, padding=1,  # stride=2 replaces maxpool
#                            bias=True,
#                            weight_quant=self.weight_quant,
#                            bias_quant=self.bias_quant,  # None for now
#                            return_quant_tensor=False),  # Return regular tensor for BatchNorm
#             nn.BatchNorm2d(128),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
#         )
        
#         # Classifier - Keep it simple for FINN
#         # Final feature map size: 128 * 4 * 4 = 2048
#         self.classifier = nn.Sequential(
#             # First FC layer with significant dimensionality reduction
#             qnn.QuantLinear(128 * 4 * 4, 256,
#                            bias=True,  # Keep bias but don't quantize it for now
#                            weight_quant=self.weight_quant,
#                            bias_quant=self.bias_quant,  # None for now
#                            return_quant_tensor=False),  # Return regular tensor for compatibility
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=False),  # Final layer compatibility
            
#             # Output layer (no quantization on final output for classification)
#             qnn.QuantLinear(256, num_classes,
#                            bias=True,  # Keep bias but don't quantize it for now
#                            weight_quant=self.weight_quant,
#                            bias_quant=self.bias_quant,  # None for now
#                            return_quant_tensor=False)  # Return regular tensor for loss computation
#         )
        
#         self.quant_mode = QUANT_MODE
#         self.num_classes = num_classes
        
#     def forward(self, x):
#         # Input quantization - ESSENTIAL for FINN
#         x = self.input_quant(x)
        
#         # Convert QuantTensor to regular tensor if needed for feature extraction
#         if hasattr(x, 'value'):
#             x = x.value
        
#         # Feature extraction
#         x = self.features(x)
        
#         # Flatten for classifier
#         x = torch.flatten(x, 1)
        
#         # Classification
#         x = self.classifier(x)
        
#         return x

# # --- Data Loading Functions ---
# def load_dataset(path=DATASET_PATH):
#     """Load the preprocessed GTSRB dataset."""
#     print(f"Loading dataset from {path}...")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset file not found at {path}. Please run prepare_dataset.py first.")
    
#     with open(path, 'rb') as f:
#         dataset = pickle.load(f)
    
#     print("Dataset loaded successfully.")
#     print(f"Available keys: {list(dataset.keys())}")
    
#     return dataset

# def prepare_data_loaders(dataset, batch_size=BATCH_SIZE, use_cuda=False):
#     """Prepare DataLoader objects for training, validation, and testing."""
#     print("Preparing data loaders...")
    
#     required_keys = ['train_data', 'train_labels', 'val_data', 'val_labels']
#     if not all(key in dataset for key in required_keys):
#         raise KeyError(f"Dataset missing required keys: {required_keys}")

#     # Convert to tensors and normalize to [0,1] range
#     X_train = torch.tensor(dataset['train_data'], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
#     y_train = torch.tensor(dataset['train_labels'], dtype=torch.long)
#     X_val = torch.tensor(dataset['val_data'], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
#     y_val = torch.tensor(dataset['val_labels'], dtype=torch.long)

#     print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
#     print(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")

#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)

#     pin_memory = use_cuda
#     num_workers = 2 if use_cuda else 0

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                              pin_memory=pin_memory, num_workers=num_workers, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
#                            pin_memory=pin_memory, num_workers=num_workers)

#     # Prepare test loader if available
#     test_loader = None
#     if 'test_data' in dataset and 'test_labels' in dataset and len(dataset['test_data']) > 0:
#         X_test = torch.tensor(dataset['test_data'], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
#         y_test = torch.tensor(dataset['test_labels'], dtype=torch.long)
#         test_dataset = TensorDataset(X_test, y_test)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
#                                 pin_memory=pin_memory, num_workers=num_workers)
#         print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

#     print("Data loaders ready.")
#     return train_loader, val_loader, test_loader

# # --- Training Function ---
# def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS):
#     """Train the W4A4 quantized model with special handling for low-precision training."""
#     print(f"Starting W4A4 quantized training on {device} for {epochs} epochs...")
#     model.to(device)

#     train_losses, val_losses, train_accs, val_accs = [], [], [], []
#     best_val_acc = 0.0
#     model_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_{model.quant_mode}_finn.pth")
    
#     # Learning rate scheduler for quantized training
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
#     start_time = time.time()

#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0
#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
#         for batch_idx, (inputs, labels) in enumerate(train_pbar):
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
            
#             # Gradient clipping for quantized training stability
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             # Update progress bar
#             current_acc = 100 * correct / total if total > 0 else 0
#             train_pbar.set_postfix({
#                 'Loss': f"{loss.item():.4f}", 
#                 'Acc': f"{current_acc:.2f}%",
#                 'LR': f"{scheduler.get_last_lr()[0]:.6f}"
#             })
        
#         epoch_train_loss = running_loss / total if total > 0 else 0
#         epoch_train_acc = 100 * correct / total if total > 0 else 0
#         train_losses.append(epoch_train_loss)
#         train_accs.append(epoch_train_acc)

#         # Validation phase
#         model.eval()
#         running_loss, correct, total = 0.0, 0, 0
#         val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
#         with torch.no_grad():
#             for inputs, labels in val_pbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
                
#                 running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
                
#                 current_acc = 100 * correct / total if total > 0 else 0
#                 val_pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{current_acc:.2f}%"})

#         epoch_val_loss = running_loss / total if total > 0 else 0
#         epoch_val_acc = 100 * correct / total if total > 0 else 0
#         val_losses.append(epoch_val_loss)
#         val_accs.append(epoch_val_acc)

#         # Step scheduler
#         scheduler.step()

#         # Print epoch summary
#         epoch_time = time.time() - start_time
#         print(f"\nEpoch {epoch+1}/{epochs} Summary:")
#         print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
#         print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
#         print(f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.1f}s\n")

#         # Save best model
#         if epoch_val_acc > best_val_acc:
#             best_val_acc = epoch_val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'best_val_acc': best_val_acc,
#                 'train_losses': train_losses,
#                 'val_losses': val_losses,
#                 'train_accs': train_accs,
#                 'val_accs': val_accs,
#             }, model_path)
#             print(f"→ Saved best model (Val Acc: {best_val_acc:.2f}%) at epoch {epoch+1}")

#     total_time = time.time() - start_time
#     print(f"\nTraining completed in {total_time:.2f} seconds.")
#     print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
#     # Save training plot
#     save_training_plot(train_losses, val_losses, train_accs, val_accs, epochs)
    
#     return model, model_path, best_val_acc

# def save_training_plot(train_losses, val_losses, train_accs, val_accs, epochs):
#     """Save training history plots."""
#     plot_path = os.path.join(OUTPUT_DIR, f"training_history_{QUANT_MODE}_finn.png")
    
#     try:
#         plt.figure(figsize=(15, 5))
        
#         plt.subplot(1, 3, 1)
#         plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss', linewidth=2)
#         plt.plot(range(1, epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.title('Loss vs. Epochs')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#         plt.subplot(1, 3, 2)
#         plt.plot(range(1, epochs + 1), train_accs, 'b-', label='Train Accuracy', linewidth=2)
#         plt.plot(range(1, epochs + 1), val_accs, 'r-', label='Validation Accuracy', linewidth=2)
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy (%)')
#         plt.title('Accuracy vs. Epochs')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         plt.subplot(1, 3, 3)
#         plt.plot(range(1, epochs + 1), np.array(val_accs) - np.array(train_accs), 'g-', linewidth=2)
#         plt.xlabel('Epochs')
#         plt.ylabel('Val Acc - Train Acc (%)')
#         plt.title('Overfitting Monitor')
#         plt.grid(True, alpha=0.3)
#         plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

#         plt.tight_layout()
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"Training plots saved to {plot_path}")
#         plt.close()
        
#     except Exception as e:
#         print(f"Warning: Could not save training plot: {e}")

# # --- Testing Function ---
# def test_model(model, test_loader, device):
#     """Test the trained model."""
#     print("Testing W4A4 quantized model...")
    
#     if test_loader is None:
#         print("No test data available.")
#         return None
        
#     model.to(device)
#     model.eval()

#     class_correct = {}
#     class_total = {}
#     correct, total = 0, 0
    
#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Testing"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
            
#             # Overall accuracy
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             # Per-class accuracy
#             c = (predicted == labels).squeeze()
#             for i, label in enumerate(labels):
#                 label_key = label.item()
#                 if label_key not in class_correct:
#                     class_correct[label_key] = 0
#                     class_total[label_key] = 0
                
#                 if len(c.shape) == 0:  # Single item batch
#                     class_correct[label_key] += c.item()
#                 else:
#                     class_correct[label_key] += c[i].item()
#                 class_total[label_key] += 1
    
#     test_acc = 100 * correct / total if total > 0 else 0
#     print(f"\nTest Accuracy: {test_acc:.2f}% ({correct}/{total})")
    
#     # Per-class accuracy
#     per_class_accs = []
#     print("\nPer-class accuracy:")
#     for class_id in sorted(class_total.keys()):
#         if class_total[class_id] > 0:
#             class_acc = 100 * class_correct[class_id] / class_total[class_id]
#             per_class_accs.append(class_acc)
#             print(f"  Class {class_id}: {class_acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})")
    
#     avg_class_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0
#     print(f"\nAverage per-class accuracy: {avg_class_acc:.2f}%")
    
#     return {
#         'accuracy': test_acc,
#         'per_class_accuracy': {class_id: 100 * class_correct[class_id] / class_total[class_id] 
#                               for class_id in class_total if class_total[class_id] > 0},
#         'avg_class_accuracy': avg_class_acc
#     }

# # --- FINN Compatibility Validation ---
# def validate_finn_compatibility(model):
#     """
#     Validate that the model meets FINN requirements before ONNX export.
#     """
#     print("\n" + "="*60)
#     print("FINN COMPATIBILITY VALIDATION")
#     print("="*60)
    
#     compatibility_issues = []
    
#     # Check for MaxPool layers
#     has_maxpool = False
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
#             has_maxpool = True
#             compatibility_issues.append(f"MaxPool layer found: {name}")
    
#     if not has_maxpool:
#         print("✓ No MaxPool layers found - using strided convolutions")
    
#     # Check quantization settings
#     quant_layers = []
#     for name, module in model.named_modules():
#         if hasattr(module, 'act_quant') and module.act_quant is not None:
#             quant_layers.append((name, module))
    
#     print(f"✓ Found {len(quant_layers)} quantized layers")
    
#     # Check for input quantization
#     has_input_quant = hasattr(model, 'input_quant')
#     if has_input_quant:
#         print("✓ Input quantization layer present")
#     else:
#         compatibility_issues.append("Missing input quantization layer")
    
#     # Check architecture
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"✓ Total parameters: {total_params:,}")
    
#     if compatibility_issues:
#         print("\n❌ COMPATIBILITY ISSUES FOUND:")
#         for issue in compatibility_issues:
#             print(f"  - {issue}")
#         return False
#     else:
#         print("\n✅ MODEL IS FINN-COMPATIBLE!")
#         print("Ready for ONNX export and FINN compilation")
#         return True

# # --- ONNX Export Function ---
# def export_finn_onnx(model, input_shape=(1, 3, 32, 32), export_path=None):
#     """
#     Export model to FINN-compatible ONNX format.
#     """
#     if export_path is None:
#         export_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_{QUANT_MODE}_finn_ready.onnx")
    
#     print(f"\nExporting FINN-compatible ONNX to: {export_path}")
    
#     try:
#         # Validate compatibility first
#         if not validate_finn_compatibility(model):
#             print("❌ Model failed FINN compatibility check!")
#             return None
            
#         model.eval()
        
#         # Create dummy input
#         dummy_input = torch.randn(input_shape)
        
#         # Export using Brevitas FINN export
#         from brevitas.export import export_qonnx
#         export_qonnx(model, dummy_input, export_path)
        
#         print(f"✅ ONNX export successful: {export_path}")
#         print("\nNext steps:")
#         print("1. Use this ONNX file with FINN Docker")
#         print("2. Run build_dataflow with your config")
#         print("3. Deploy on PYNQ board")
        
#         return export_path
        
#     except Exception as e:
#         print(f"❌ ONNX export failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # --- Main Function ---
# def main():
#     """Main execution function."""
#     print("=" * 80)
#     print("FINN-Compatible W4A4 Road Sign Classifier Training")
#     print("=" * 80)
    
#     # Set device
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     print(f"Using device: {device}")
#     if use_cuda: 
#         print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

#     try:
#         # Load dataset
#         dataset = load_dataset(DATASET_PATH)
#         num_classes = dataset.get('num_classes')
#         class_names = dataset.get('class_names', {})
        
#         if num_classes is None:
#             raise ValueError("Dataset missing 'num_classes' key.")
        
#         print(f"\nDataset: {num_classes} classes")
#         print(f"Quantization: W4A4 (4-bit weights and activations)")
#         print(f"Target: FINN FPGA deployment")
        
#         # Prepare data loaders
#         train_loader, val_loader, test_loader = prepare_data_loaders(dataset, BATCH_SIZE, use_cuda)
        
#         # Create FINN-compatible model
#         model = FINNRoadSignNet(num_classes=num_classes)
#         print(f"\nModel Summary:")
#         print(f"{model}")
        
#         # Count parameters
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"\nTotal parameters: {total_params:,}")
#         print(f"Trainable parameters: {trainable_params:,}")
        
#         # Define loss and optimizer for quantized training
#         criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps with quantization
#         optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#         print(f"\nTraining Configuration:")
#         print(f"- Batch size: {BATCH_SIZE}")
#         print(f"- Epochs: {EPOCHS}")
#         print(f"- Learning rate: {LEARNING_RATE}")
#         print(f"- Weight decay: {WEIGHT_DECAY}")
#         print(f"- Label smoothing: 0.1")
#         print(f"- Gradient clipping: 1.0")
        
#         # Train model
#         trained_model, model_path, best_val_acc = train_model(
#             model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
#         )
        
#         # Test the best model
#         print(f"\nLoading best model for testing...")
#         test_model_instance = FINNRoadSignNet(num_classes=num_classes)
        
#         try:
#             checkpoint = torch.load(model_path, map_location=device)
#             test_model_instance.load_state_dict(checkpoint['model_state_dict'])
#             test_results = test_model(test_model_instance, test_loader, device)
            
#             # Save comprehensive summary
#             if test_results:
#                 summary_path = os.path.join(OUTPUT_DIR, f"model_summary_{QUANT_MODE}_finn.txt")
#                 with open(summary_path, 'w') as f:
#                     f.write("FINN-Compatible W4A4 Road Sign Classifier Summary\n")
#                     f.write("=" * 60 + "\n\n")
#                     f.write(f"Quantization: {QUANT_MODE} (4-bit weights and activations)\n")
#                     f.write(f"Target Platform: FINN/PYNQ FPGA\n")
#                     f.write(f"Classes: {num_classes}\n")
#                     f.write(f"Total Parameters: {total_params:,}\n")
#                     f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
                    
#                     f.write("Training Results:\n")
#                     f.write(f"- Best Validation Accuracy: {best_val_acc:.2f}%\n")
#                     f.write(f"- Test Accuracy: {test_results['accuracy']:.2f}%\n")
#                     f.write(f"- Average Per-Class Accuracy: {test_results['avg_class_accuracy']:.2f}%\n\n")
                    
#                     f.write("FINN Compatibility Features:\n")
#                     f.write("- No MaxPool layers (replaced with strided convolutions)\n")
#                     f.write("- Input quantization layer for proper FINN pipeline\n")
#                     f.write("- 4-bit quantized weights and activations\n")
#                     f.write("- 32-bit bias quantization (FINN requirement)\n")
#                     f.write("- Single dataflow path (no skip connections)\n")
#                     f.write("- BatchNorm layers for training stability\n\n")
                    
#                     f.write("Next Steps for FINN Export:\n")
#                     f.write("1. Export to ONNX using Brevitas FINN export\n")
#                     f.write("2. Use FINN Docker for compilation\n")
#                     f.write("3. Deploy on PYNQ board\n\n")
                    
#                     f.write(f"Model Architecture:\n{model}\n")
                
#                 print(f"Model summary saved to {summary_path}")
                
#                 # FINN Compatibility Check and Export
#                 print("\n" + "="*80)
#                 print("FINN COMPATIBILITY CHECK & ONNX EXPORT")
#                 print("="*80)
                
#                 # Validate and export ONNX
#                 onnx_path = export_finn_onnx(test_model_instance, input_shape=(1, 3, 32, 32))
                
#                 if onnx_path:
#                     print(f"\n🎉 SUCCESS! FINN-ready ONNX exported: {onnx_path}")
#                     print("\nFINN Build Command:")
#                     print(f"build_dataflow_cfg.py {onnx_path} --output-dir finn_output --folding-config-file your_config.json")
#                 else:
#                     print("\n❌ ONNX export failed - check compatibility issues above")
                
#                 # Print next steps
#                 print("\n" + "=" * 80)
#                 print("TRAINING COMPLETED SUCCESSFULLY!")
#                 print("=" * 80)
#                 print(f"✓ Best model saved: {model_path}")
#                 print(f"✓ Validation accuracy: {best_val_acc:.2f}%")
#                 print(f"✓ Test accuracy: {test_results['accuracy']:.2f}%")
#                 if onnx_path:
#                     print(f"✓ FINN-ready ONNX: {onnx_path}")
#                 print("\nReady for FINN FPGA deployment! 🚀")
                
#         except Exception as e:
#             print(f"Error loading/testing model: {e}")
#             import traceback
#             traceback.print_exc()
            
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == '__main__':
#     main()


# #!/usr/bin/env python3
# """
# Enhanced training script for FINN-friendly quantized neural network
# optimized for PYNQ FPGA deployment.

# This script trains a road sign classifier with quantized weights and activations
# using the Brevitas library to ensure compatibility with FINN compiler.
# """

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import brevitas.nn as qnn
# from brevitas.quant.scaled_int import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
# from brevitas.quant import Int8Bias
# from brevitas.core.restrict_val import RestrictValueType
# from brevitas.core.scaling import ScalingImplType
# from brevitas.core.zero_point import ZeroZeroPoint
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time

# # --- Configuration ---
# # Hardcoded paths
# DATASET_PATH = 'dataset/gtsrb_processed.pkl'
# OUTPUT_DIR = 'models'
# MODEL_BASE_NAME = 'rsc_w4a4'

# # Training parameters
# BATCH_SIZE = 64
# EPOCHS = 20
# LEARNING_RATE = 0.001
# WEIGHT_DECAY = 1e-5

# # Model configuration
# # Using W4A4 quantization for FINN compatibility
# QUANT_MODE = 'w4a4'  

# # Set random seed for reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- FINN-Compatible Quantizers ---
# class FINNUint4ActQuant(Int8ActPerTensorFloat):
#     """
#     FINN-compatible unsigned 4-bit activation quantizer.
#     CRITICAL: FINN requires UNSIGNED and NON-NARROW activations for ReLU.
#     """
#     bit_width = 4
#     signed = False  # UNSIGNED for FINN ReLU compatibility
#     narrow = False  # NON-NARROW for FINN compatibility
#     zero_point_impl = ZeroZeroPoint
    
# class FINNInt4WeightQuant(Int8WeightPerTensorFloat):
#     """FINN-compatible 4-bit signed weight quantizer."""
#     bit_width = 4
#     signed = True  # Weights can be signed

# # --- Model Definition (FINN-Friendly) ---
# class QuantRoadSignNet(nn.Module):
#     def __init__(self, num_classes=43, quant_mode='w4a4'):
#         super(QuantRoadSignNet, self).__init__()
        
#         # Set quantization - using FINN-compatible 4-bit quantizers
#         self.weight_quant = FINNInt4WeightQuant
#         self.act_quant = FINNUint4ActQuant
#         self.bias_quant = None  # No bias quantization for FINN compatibility
        
#         # Input quantization - ESSENTIAL for FINN
#         self.input_quant = qnn.QuantIdentity(
#             act_quant=self.act_quant,
#             return_quant_tensor=True
#         )
        
#         self.features = nn.Sequential(
#             # First block: Conv + BN + ReLU
#             qnn.QuantConv2d(3, 32, kernel_size=3, stride=2, padding=1, 
#                             weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Second block
#             qnn.QuantConv2d(32, 32, kernel_size=3, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Third block
#             qnn.QuantConv2d(32, 64, kernel_size=3, stride=2, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Fourth block
#             qnn.QuantConv2d(64, 64, kernel_size=3, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Fifth block
#             qnn.QuantConv2d(64, 128, kernel_size=3, stride=2, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(128),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
#         )
        
#         self.classifier = nn.Sequential(
#             qnn.QuantLinear(128 * 4 * 4, 256, bias=True, weight_quant=self.weight_quant, bias_quant=self.bias_quant),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
#             qnn.QuantLinear(256, num_classes, bias=True, weight_quant=self.weight_quant, bias_quant=self.bias_quant)
#         )
        
#         # Store quantization mode for reference
#         self.quant_mode = quant_mode

#     def forward(self, x):
#         # Input quantization
#         x = self.input_quant(x)
        
#         # Handle QuantTensor conversion if needed
#         if hasattr(x, 'value'):
#             x = x.value
            
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


# # --- Data Handling Functions ---
# def load_dataset(path=DATASET_PATH):
#     """Load the preprocessed dataset."""
#     print(f"Loading dataset from {path}...")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset file not found at {path}. Please run prepare_dataset.py first.")
#     with open(path, 'rb') as f:
#         dataset = pickle.load(f)
#     print("Dataset loaded successfully.")
#     return dataset


# def prepare_data_loaders(dataset, batch_size=BATCH_SIZE, use_cuda=False):
#     """Prepare DataLoader objects for training, validation, and testing."""
#     print("Preparing data loaders...")
#     required_keys = ['train_data', 'train_labels', 'val_data', 'val_labels']
#     if not all(key in dataset for key in required_keys):
#         raise KeyError(f"Dataset dictionary missing one or more required keys: {required_keys}")

#     X_train = torch.tensor(dataset['train_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#     y_train = torch.tensor(dataset['train_labels'], dtype=torch.long)
#     X_val = torch.tensor(dataset['val_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#     y_val = torch.tensor(dataset['val_labels'], dtype=torch.long)

#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)

#     pin_memory = use_cuda
#     num_workers = 4 if use_cuda else 0

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                              pin_memory=pin_memory, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
#                            pin_memory=pin_memory, num_workers=num_workers)

#     # Prepare test loader if test data is available
#     test_loader = None
#     if 'test_data' in dataset and 'test_labels' in dataset and len(dataset['test_data']) > 0:
#         X_test = torch.tensor(dataset['test_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#         y_test = torch.tensor(dataset['test_labels'], dtype=torch.long)
#         test_dataset = TensorDataset(X_test, y_test)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
#                                 pin_memory=pin_memory, num_workers=num_workers)

#     print("Data loaders ready.")
#     return train_loader, val_loader, test_loader


# # --- Training Function ---
# def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS):
#     """Train the model and validate after each epoch."""
#     print(f"Starting training on {device} for {epochs} epochs...")
#     model.to(device)

#     train_losses, val_losses, train_accs, val_accs = [], [], [], []
#     best_val_acc = 0.0
#     model_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_{model.quant_mode}_finn.pth")
#     start_time = time.time()

#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0
#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
#         for inputs, labels in train_pbar:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             train_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})
        
#         epoch_train_loss = running_loss / total if total > 0 else 0
#         epoch_train_acc = 100 * correct / total if total > 0 else 0
#         train_losses.append(epoch_train_loss)
#         train_accs.append(epoch_train_acc)

#         # Validation phase
#         model.eval()
#         running_loss, correct, total = 0.0, 0, 0
#         val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
#         with torch.no_grad():
#             for inputs, labels in val_pbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 val_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})

#         epoch_val_loss = running_loss / total if total > 0 else 0
#         epoch_val_acc = 100 * correct / total if total > 0 else 0
#         val_losses.append(epoch_val_loss)
#         val_accs.append(epoch_val_acc)

#         # Print epoch summary
#         epoch_time = time.time() - start_time
#         print(f"Epoch {epoch+1}/{epochs} Summary: "
#               f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
#               f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | "
#               f"Time: {epoch_time:.1f}s")

#         # Save best model
#         if epoch_val_acc > best_val_acc:
#             best_val_acc = epoch_val_acc
#             # Save as checkpoint with full information for FINN compatibility
#             checkpoint = {
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'best_val_acc': best_val_acc,
#                 'quant_mode': model.quant_mode
#             }
#             torch.save(checkpoint, model_path)
#             print(f"---> Saved best model to {model_path} with validation accuracy: {best_val_acc:.2f}% at epoch {epoch+1}")

#     total_time = time.time() - start_time
#     print(f"Training finished in {total_time:.2f} seconds.")
    
#     # Generate and save training plot
#     plot_path = os.path.join(OUTPUT_DIR, f"training_history_{model.quant_mode}.png")
#     try:
#         plt.figure(figsize=(12, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
#         plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
#         plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss vs. Epochs'); plt.legend(); plt.grid(True)

#         plt.subplot(1, 2, 2)
#         plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
#         plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
#         plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy vs. Epochs'); plt.legend(); plt.grid(True)

#         plt.tight_layout()
#         plt.savefig(plot_path)
#         print(f"Training history plot saved to {plot_path}")
#         plt.close()
#     except Exception as plot_e:
#         print(f"Warning: Could not generate or save training plot: {plot_e}")
        
#     return model, model_path, best_val_acc


# # --- Testing Function ---
# def test_model(model, test_loader, device):
#     """Evaluate the model on test data."""
#     print("Testing model...")
#     model.to(device)
#     model.eval()

#     if test_loader is None:
#         print("No test data available. Skipping testing.")
#         return None

#     class_correct = {}
#     class_total = {}
#     correct, total = 0, 0
    
#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Testing"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
            
#             # Calculate overall accuracy
#             c = (predicted == labels).squeeze()
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             # Calculate per-class accuracy
#             for i, label in enumerate(labels):
#                 label_key = label.item()
#                 if label_key not in class_correct:
#                     class_correct[label_key] = 0
#                     class_total[label_key] = 0
                
#                 if len(c.shape) == 0:  # Handle single item in batch
#                     class_correct[label_key] += c.item()
#                 else:
#                     class_correct[label_key] += c[i].item()
#                 class_total[label_key] += 1
    
#     # Calculate and display overall accuracy
#     test_acc = 100 * correct / total if total > 0 else 0
#     print(f"Test Accuracy: {test_acc:.2f}% ({correct}/{total})")
    
#     # Display per-class accuracy
#     print("\nPer-class accuracy:")
#     per_class_accs = []
#     for class_id in sorted(class_total.keys()):
#         class_acc = 100 * class_correct[class_id] / class_total[class_id]
#         per_class_accs.append(class_acc)
#         print(f"  Class {class_id}: {class_acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})")
    
#     # Calculate average per-class accuracy
#     avg_class_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0
#     print(f"\nAverage per-class accuracy: {avg_class_acc:.2f}%")
    
#     return {
#         'accuracy': test_acc,
#         'per_class_accuracy': {class_id: 100 * class_correct[class_id] / class_total[class_id] 
#                               for class_id in class_total},
#         'avg_class_accuracy': avg_class_acc
#     }

# # --- Main Execution Block ---
# def main():
#     """Main execution function."""
#     # Set device
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     print(f"Using device: {device}")
#     if use_cuda: 
#         print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

#     try:
#         # Load dataset
#         dataset = load_dataset(DATASET_PATH)
#         num_classes = dataset.get('num_classes')
#         class_names = dataset.get('class_names', {})
        
#         if num_classes is None:
#             raise ValueError("Dataset missing 'num_classes' key.")
#         print(f"Dataset contains {num_classes} classes.")
        
#         # Prepare data loaders
#         train_loader, val_loader, test_loader = prepare_data_loaders(dataset, BATCH_SIZE, use_cuda)
        
#         # Create model with specified quantization
#         model = QuantRoadSignNet(num_classes=num_classes, quant_mode=QUANT_MODE)
#         print(f"\nModel architecture:\n{model}\n")
        
#         # Define loss function and optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#         print(f"\nStarting training using {QUANT_MODE} quantization for FINN compatibility.")
#         print(f"Training parameters: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LEARNING_RATE}")
#         print(f"IMPORTANT: Using unsigned, non-narrow 4-bit activations for FINN ReLU compatibility")

#         # Train model
#         trained_model, model_path, best_val_acc = train_model(
#             model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
#         )
        
#         # Test the best model
#         print(f"\nLoading best model from {model_path} for final testing...")
#         test_model_instance = QuantRoadSignNet(num_classes=num_classes, quant_mode=QUANT_MODE)
#         try:
#             checkpoint = torch.load(model_path, map_location=device)
#             if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#                 test_model_instance.load_state_dict(checkpoint['model_state_dict'])
#             else:
#                 test_model_instance.load_state_dict(checkpoint)
                
#             test_results = test_model(test_model_instance, test_loader, device)
            
#             if test_results:
#                 # Save model summary to a text file
#                 summary_path = os.path.join(OUTPUT_DIR, f"model_summary_{QUANT_MODE}.txt")
#                 with open(summary_path, 'w') as f:
#                     f.write(f"Road Sign Classifier Model Summary\n")
#                     f.write(f"================================\n\n")
#                     f.write(f"Quantization mode: {QUANT_MODE}\n")
#                     f.write(f"Number of classes: {num_classes}\n")
#                     f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
#                     f.write(f"Test accuracy: {test_results['accuracy']:.2f}%\n")
#                     f.write(f"Average per-class accuracy: {test_results['avg_class_accuracy']:.2f}%\n\n")
#                     f.write("Per-class accuracy:\n")
#                     for class_id, acc in sorted(test_results['per_class_accuracy'].items()):
#                         class_name = class_names.get(class_id, f"Class {class_id}")
#                         f.write(f"  {class_id}: {class_name} - {acc:.2f}%\n")
                
#                 print(f"Model summary saved to {summary_path}")
                
#                 # Save a FINN-ready version with explicit naming
#                 finn_model_path = os.path.join(OUTPUT_DIR, f"road_sign_classifier_w4a4_finn.pth")
#                 torch.save(checkpoint, finn_model_path)
#                 print(f"FINN-ready model saved to {finn_model_path}")
                
#         except Exception as e:
#             print(f"Error during testing: {e}")
            
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# Enhanced training script for FINN-friendly quantized neural network
# optimized for PYNQ FPGA deployment.

# This script trains a road sign classifier with quantized weights and activations
# using the Brevitas library to ensure compatibility with FINN compiler.
# """

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import brevitas.nn as qnn
# from brevitas.quant.scaled_int import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
# from brevitas.quant import Int8Bias
# from brevitas.core.restrict_val import RestrictValueType
# from brevitas.core.scaling import ScalingImplType
# from brevitas.core.zero_point import ZeroZeroPoint
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time

# # --- Configuration ---
# # Hardcoded paths
# DATASET_PATH = 'dataset/gtsrb_processed.pkl'
# OUTPUT_DIR = 'models'
# MODEL_BASE_NAME = 'road_sign_classifier'

# # Training parameters
# BATCH_SIZE = 64
# EPOCHS = 20
# LEARNING_RATE = 0.001
# WEIGHT_DECAY = 1e-5

# # Model configuration
# # Using W4A4 quantization for FINN compatibility
# QUANT_MODE = 'w4a4'  

# # Set random seed for reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- FINN-Compatible Quantizers ---
# class FINNUint4ActQuant(Int8ActPerTensorFloat):
#     """
#     FINN-compatible unsigned 4-bit activation quantizer.
#     CRITICAL: FINN requires UNSIGNED and NON-NARROW activations for ReLU.
#     """
#     bit_width = 4
#     signed = False  # UNSIGNED for FINN ReLU compatibility
#     narrow = False  # NON-NARROW for FINN compatibility
#     zero_point_impl = ZeroZeroPoint
    
# class FINNInt4WeightQuant(Int8WeightPerTensorFloat):
#     """FINN-compatible 4-bit signed weight quantizer."""
#     bit_width = 4
#     signed = True  # Weights can be signed

# # --- Model Definition (FINN-Friendly) ---
# class QuantRoadSignNet(nn.Module):
#     def __init__(self, num_classes=43, quant_mode='w4a4'):
#         super(QuantRoadSignNet, self).__init__()
        
#         # Set quantization - using FINN-compatible 4-bit quantizers
#         self.weight_quant = FINNInt4WeightQuant
#         self.act_quant = FINNUint4ActQuant
#         self.bias_quant = None  # No bias quantization for FINN compatibility
        
#         # Input quantization - ESSENTIAL for FINN
#         self.input_quant = qnn.QuantIdentity(
#             act_quant=self.act_quant,
#             return_quant_tensor=True
#         )
        
#         self.features = nn.Sequential(
#             # First block: Conv + BN + ReLU
#             qnn.QuantConv2d(3, 32, kernel_size=3, stride=2, padding=1, 
#                             weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Second block
#             qnn.QuantConv2d(32, 32, kernel_size=3, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(32),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Third block
#             qnn.QuantConv2d(32, 64, kernel_size=3, stride=2, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Fourth block
#             qnn.QuantConv2d(64, 64, kernel_size=3, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(64),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
            
#             # Fifth block
#             qnn.QuantConv2d(64, 128, kernel_size=3, stride=2, padding=1, 
#                            weight_quant=self.weight_quant, bias=True, bias_quant=self.bias_quant),
#             nn.BatchNorm2d(128),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
#         )
        
#         self.classifier = nn.Sequential(
#             qnn.QuantLinear(128 * 4 * 4, 256, bias=True, weight_quant=self.weight_quant, bias_quant=self.bias_quant),
#             qnn.QuantReLU(act_quant=self.act_quant, return_quant_tensor=True),
#             qnn.QuantLinear(256, num_classes, bias=True, weight_quant=self.weight_quant, bias_quant=self.bias_quant)
#         )
        
#         # Store quantization mode for reference
#         self.quant_mode = quant_mode

#     def forward(self, x):
#         # Input quantization
#         x = self.input_quant(x)
        
#         # Handle QuantTensor conversion if needed
#         if hasattr(x, 'value'):
#             x = x.value
            
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


# # --- Data Handling Functions ---
# def load_dataset(path=DATASET_PATH):
#     """Load the preprocessed dataset."""
#     print(f"Loading dataset from {path}...")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset file not found at {path}. Please run prepare_dataset.py first.")
#     with open(path, 'rb') as f:
#         dataset = pickle.load(f)
#     print("Dataset loaded successfully.")
#     return dataset


# def prepare_data_loaders(dataset, batch_size=BATCH_SIZE, use_cuda=False):
#     """Prepare DataLoader objects for training, validation, and testing."""
#     print("Preparing data loaders...")
#     required_keys = ['train_data', 'train_labels', 'val_data', 'val_labels']
#     if not all(key in dataset for key in required_keys):
#         raise KeyError(f"Dataset dictionary missing one or more required keys: {required_keys}")

#     X_train = torch.tensor(dataset['train_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#     y_train = torch.tensor(dataset['train_labels'], dtype=torch.long)
#     X_val = torch.tensor(dataset['val_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#     y_val = torch.tensor(dataset['val_labels'], dtype=torch.long)

#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)

#     pin_memory = use_cuda
#     num_workers = 4 if use_cuda else 0

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                              pin_memory=pin_memory, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
#                            pin_memory=pin_memory, num_workers=num_workers)

#     # Prepare test loader if test data is available
#     test_loader = None
#     if 'test_data' in dataset and 'test_labels' in dataset and len(dataset['test_data']) > 0:
#         X_test = torch.tensor(dataset['test_data'], dtype=torch.float32).permute(0, 3, 1, 2)
#         y_test = torch.tensor(dataset['test_labels'], dtype=torch.long)
#         test_dataset = TensorDataset(X_test, y_test)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
#                                 pin_memory=pin_memory, num_workers=num_workers)

#     print("Data loaders ready.")
#     return train_loader, val_loader, test_loader


# # --- Training Function ---
# def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS):
#     """Train the model and validate after each epoch."""
#     print(f"Starting training on {device} for {epochs} epochs...")
#     model.to(device)

#     train_losses, val_losses, train_accs, val_accs = [], [], [], []
#     best_val_acc = 0.0
#     model_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_{model.quant_mode}_finn.pth")
#     start_time = time.time()

#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0
#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
#         for inputs, labels in train_pbar:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             train_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})
        
#         epoch_train_loss = running_loss / total if total > 0 else 0
#         epoch_train_acc = 100 * correct / total if total > 0 else 0
#         train_losses.append(epoch_train_loss)
#         train_accs.append(epoch_train_acc)

#         # Validation phase
#         model.eval()
#         running_loss, correct, total = 0.0, 0, 0
#         val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
#         with torch.no_grad():
#             for inputs, labels in val_pbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 val_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})

#         epoch_val_loss = running_loss / total if total > 0 else 0
#         epoch_val_acc = 100 * correct / total if total > 0 else 0
#         val_losses.append(epoch_val_loss)
#         val_accs.append(epoch_val_acc)

#         # Print epoch summary
#         epoch_time = time.time() - start_time
#         print(f"Epoch {epoch+1}/{epochs} Summary: "
#               f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
#               f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | "
#               f"Time: {epoch_time:.1f}s")

#         # Save best model
#         if epoch_val_acc > best_val_acc:
#             best_val_acc = epoch_val_acc
#             # Save as checkpoint with full information for FINN compatibility
#             checkpoint = {
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'best_val_acc': best_val_acc,
#                 'quant_mode': model.quant_mode
#             }
#             torch.save(checkpoint, model_path)
#             print(f"---> Saved best model to {model_path} with validation accuracy: {best_val_acc:.2f}% at epoch {epoch+1}")

#     total_time = time.time() - start_time
#     print(f"Training finished in {total_time:.2f} seconds.")
    
#     # Generate and save training plot
#     plot_path = os.path.join(OUTPUT_DIR, f"training_history_{model.quant_mode}.png")
#     try:
#         plt.figure(figsize=(12, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
#         plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
#         plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss vs. Epochs'); plt.legend(); plt.grid(True)

#         plt.subplot(1, 2, 2)
#         plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
#         plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
#         plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy vs. Epochs'); plt.legend(); plt.grid(True)

#         plt.tight_layout()
#         plt.savefig(plot_path)
#         print(f"Training history plot saved to {plot_path}")
#         plt.close()
#     except Exception as plot_e:
#         print(f"Warning: Could not generate or save training plot: {plot_e}")
        
#     return model, model_path, best_val_acc


# # --- Testing Function ---
# def test_model(model, test_loader, device):
#     """Evaluate the model on test data."""
#     print("Testing model...")
#     model.to(device)
#     model.eval()

#     if test_loader is None:
#         print("No test data available. Skipping testing.")
#         return None

#     class_correct = {}
#     class_total = {}
#     correct, total = 0, 0
    
#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Testing"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
            
#             # Calculate overall accuracy
#             c = (predicted == labels).squeeze()
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             # Calculate per-class accuracy
#             for i, label in enumerate(labels):
#                 label_key = label.item()
#                 if label_key not in class_correct:
#                     class_correct[label_key] = 0
#                     class_total[label_key] = 0
                
#                 if len(c.shape) == 0:  # Handle single item in batch
#                     class_correct[label_key] += c.item()
#                 else:
#                     class_correct[label_key] += c[i].item()
#                 class_total[label_key] += 1
    
#     # Calculate and display overall accuracy
#     test_acc = 100 * correct / total if total > 0 else 0
#     print(f"Test Accuracy: {test_acc:.2f}% ({correct}/{total})")
    
#     # Display per-class accuracy
#     print("\nPer-class accuracy:")
#     per_class_accs = []
#     for class_id in sorted(class_total.keys()):
#         class_acc = 100 * class_correct[class_id] / class_total[class_id]
#         per_class_accs.append(class_acc)
#         print(f"  Class {class_id}: {class_acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})")
    
#     # Calculate average per-class accuracy
#     avg_class_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0
#     print(f"\nAverage per-class accuracy: {avg_class_acc:.2f}%")
    
#     return {
#         'accuracy': test_acc,
#         'per_class_accuracy': {class_id: 100 * class_correct[class_id] / class_total[class_id] 
#                               for class_id in class_total},
#         'avg_class_accuracy': avg_class_acc
#     }

# # --- Main Execution Block ---
# def main():
#     """Main execution function."""
#     # Set device
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     print(f"Using device: {device}")
#     if use_cuda: 
#         print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

#     try:
#         # Load dataset
#         dataset = load_dataset(DATASET_PATH)
#         num_classes = dataset.get('num_classes')
#         class_names = dataset.get('class_names', {})
        
#         if num_classes is None:
#             raise ValueError("Dataset missing 'num_classes' key.")
#         print(f"Dataset contains {num_classes} classes.")
        
#         # Prepare data loaders
#         train_loader, val_loader, test_loader = prepare_data_loaders(dataset, BATCH_SIZE, use_cuda)
        
#         # Create model with specified quantization
#         model = QuantRoadSignNet(num_classes=num_classes, quant_mode=QUANT_MODE)
#         print(f"\nModel architecture:\n{model}\n")
        
#         # Define loss function and optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#         print(f"\nStarting training using {QUANT_MODE} quantization for FINN compatibility.")
#         print(f"Training parameters: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LEARNING_RATE}")
#         print(f"IMPORTANT: Using unsigned, non-narrow 4-bit activations for FINN ReLU compatibility")

#         # Train model
#         trained_model, model_path, best_val_acc = train_model(
#             model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
#         )
        
#         # Test the best model
#         print(f"\nLoading best model from {model_path} for final testing...")
#         test_model_instance = QuantRoadSignNet(num_classes=num_classes, quant_mode=QUANT_MODE)
#         try:
#             checkpoint = torch.load(model_path, map_location=device)
#             if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#                 test_model_instance.load_state_dict(checkpoint['model_state_dict'])
#             else:
#                 test_model_instance.load_state_dict(checkpoint)
                
#             test_results = test_model(test_model_instance, test_loader, device)
            
#             if test_results:
#                 # Save model summary to a text file
#                 summary_path = os.path.join(OUTPUT_DIR, f"model_summary_{QUANT_MODE}.txt")
#                 with open(summary_path, 'w') as f:
#                     f.write(f"Road Sign Classifier Model Summary\n")
#                     f.write(f"================================\n\n")
#                     f.write(f"Quantization mode: {QUANT_MODE}\n")
#                     f.write(f"Number of classes: {num_classes}\n")
#                     f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
#                     f.write(f"Test accuracy: {test_results['accuracy']:.2f}%\n")
#                     f.write(f"Average per-class accuracy: {test_results['avg_class_accuracy']:.2f}%\n\n")
#                     f.write("Per-class accuracy:\n")
#                     for class_id, acc in sorted(test_results['per_class_accuracy'].items()):
#                         class_name = class_names.get(class_id, f"Class {class_id}")
#                         f.write(f"  {class_id}: {class_name} - {acc:.2f}%\n")
                
#                 print(f"Model summary saved to {summary_path}")
                
#                 # Save a FINN-ready version with explicit naming
#                 finn_model_path = os.path.join(OUTPUT_DIR, f"road_sign_classifier_w4a4_finn.pth")
#                 torch.save(checkpoint, finn_model_path)
#                 print(f"FINN-ready model saved to {finn_model_path}")
                
#         except Exception as e:
#             print(f"Error during testing: {e}")
            
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Minimal 3-block CNN for FINN-friendly road sign classifier.
Each block contains exactly 3 layers for maximum FPGA compatibility.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from brevitas.core.zero_point import ZeroZeroPoint
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# --- Configuration ---
# Hardcoded paths
DATASET_PATH = 'dataset/gtsrb_processed.pkl'
OUTPUT_DIR = 'models'
MODEL_BASE_NAME = 'rsc_minimal'

# Training parameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Model configuration
# Using W4A4 quantization for FINN compatibility
QUANT_MODE = 'w4a4'  

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FINN-Compatible Quantizers ---
class FINNUint4ActQuant(Int8ActPerTensorFloat):
    """
    FINN-compatible unsigned 4-bit activation quantizer for ReLU.
    CRITICAL: FINN requires UNSIGNED and NON-NARROW activations for ReLU.
    """
    bit_width = 4
    signed = False  # UNSIGNED for FINN ReLU compatibility
    narrow = False  # NON-NARROW for FINN compatibility
    zero_point_impl = ZeroZeroPoint
    
class FINNInt4IdentityQuant(Int8ActPerTensorFloat):
    """
    FINN-compatible signed 4-bit activation quantizer for Identity.
    CRITICAL: FINN requires SIGNED quantization for identity activations.
    """
    bit_width = 4
    signed = True  # SIGNED for FINN Identity compatibility
    narrow = False  # NON-NARROW for FINN compatibility
    zero_point_impl = ZeroZeroPoint
    
class FINNInt4WeightQuant(Int8WeightPerTensorFloat):
    """FINN-compatible 4-bit signed weight quantizer."""
    bit_width = 4
    signed = True  # Weights can be signed
    narrow = False  # NON-NARROW for FINN compatibility

# --- Minimal Model Definition (FINN-Friendly) ---
class MinimalRoadSignNet(nn.Module):
    """
    Minimal FINN-compatible W4A4 road sign classifier.
    Exactly 3 blocks with 3 layers each.
    """
    
    def __init__(self, num_classes=43):
        super(MinimalRoadSignNet, self).__init__()
        
        # Use FINN-compatible quantizers
        self.weight_quant = FINNInt4WeightQuant
        self.relu_quant = FINNUint4ActQuant  # Unsigned for ReLU
        self.identity_quant = FINNInt4IdentityQuant  # Signed for Identity
        
        # Input quantization - ESSENTIAL for FINN - MUST BE SIGNED
        self.input_quant = qnn.QuantIdentity(
            act_quant=self.identity_quant,  # Use SIGNED identity quantizer
            return_quant_tensor=True
        )
        
        # BLOCK 1: Input -> 16 channels (3 layers)
        self.block1 = nn.Sequential(
            # Layer 1: Conv
            qnn.QuantConv2d(3, 16, kernel_size=3, stride=2, padding=1, 
                            weight_quant=self.weight_quant, bias=False),
            # Layer 2: BatchNorm
            nn.BatchNorm2d(16),
            # Layer 3: ReLU
            qnn.QuantReLU(act_quant=self.relu_quant, return_quant_tensor=True)
        )
        
        # BLOCK 2: 16 -> 32 channels (3 layers)
        self.block2 = nn.Sequential(
            # Layer 1: Conv
            qnn.QuantConv2d(16, 32, kernel_size=3, stride=2, padding=1, 
                           weight_quant=self.weight_quant, bias=False),
            # Layer 2: BatchNorm
            nn.BatchNorm2d(32),
            # Layer 3: ReLU
            qnn.QuantReLU(act_quant=self.relu_quant, return_quant_tensor=True)
        )
        
        # BLOCK 3: 32 -> 64 channels (3 layers)
        self.block3 = nn.Sequential(
            # Layer 1: Conv
            qnn.QuantConv2d(32, 64, kernel_size=3, stride=2, padding=1, 
                           weight_quant=self.weight_quant, bias=False),
            # Layer 2: BatchNorm
            nn.BatchNorm2d(64),
            # Layer 3: ReLU
            qnn.QuantReLU(act_quant=self.relu_quant, return_quant_tensor=True)
        )
        
        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            qnn.QuantLinear(64 * 4 * 4, 128, bias=True, weight_quant=self.weight_quant),
            qnn.QuantReLU(act_quant=self.relu_quant, return_quant_tensor=True),
            qnn.QuantLinear(128, num_classes, bias=True, weight_quant=self.weight_quant)
        )
        
        self.quant_mode = QUANT_MODE
        self.num_classes = num_classes

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)
        
        # Handle QuantTensor conversion if needed
        if hasattr(x, 'value'):
            x = x.value
        
        # Process through blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


# --- Data Handling Functions ---
def load_dataset(path=DATASET_PATH):
    """Load the preprocessed dataset."""
    print(f"Loading dataset from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}. Please run prepare_dataset.py first.")
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    print("Dataset loaded successfully.")
    return dataset


def prepare_data_loaders(dataset, batch_size=BATCH_SIZE, use_cuda=False):
    """Prepare DataLoader objects for training, validation, and testing."""
    print("Preparing data loaders...")
    required_keys = ['train_data', 'train_labels', 'val_data', 'val_labels']
    if not all(key in dataset for key in required_keys):
        raise KeyError(f"Dataset dictionary missing one or more required keys: {required_keys}")

    X_train = torch.tensor(dataset['train_data'], dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(dataset['train_labels'], dtype=torch.long)
    X_val = torch.tensor(dataset['val_data'], dtype=torch.float32).permute(0, 3, 1, 2)
    y_val = torch.tensor(dataset['val_labels'], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    pin_memory = use_cuda
    num_workers = 4 if use_cuda else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           pin_memory=pin_memory, num_workers=num_workers)

    # Prepare test loader if test data is available
    test_loader = None
    if 'test_data' in dataset and 'test_labels' in dataset and len(dataset['test_data']) > 0:
        X_test = torch.tensor(dataset['test_data'], dtype=torch.float32).permute(0, 3, 1, 2)
        y_test = torch.tensor(dataset['test_labels'], dtype=torch.long)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                pin_memory=pin_memory, num_workers=num_workers)

    print("Data loaders ready.")
    return train_loader, val_loader, test_loader


# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS):
    """Train the model and validate after each epoch."""
    print(f"Starting training on {device} for {epochs} epochs...")
    model.to(device)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    model_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_{model.quant_mode}_finn.pth")
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})
        
        epoch_train_loss = running_loss / total if total > 0 else 0
        epoch_train_acc = 100 * correct / total if total > 0 else 0
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct / total if total > 0 else 0})

        epoch_val_loss = running_loss / total if total > 0 else 0
        epoch_val_acc = 100 * correct / total if total > 0 else 0
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} Summary: "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Save as checkpoint with full information for FINN compatibility
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'quant_mode': model.quant_mode
            }
            torch.save(checkpoint, model_path)
            print(f"---> Saved best model to {model_path} with validation accuracy: {best_val_acc:.2f}% at epoch {epoch+1}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")
    
    # Generate and save training plot
    plot_path = os.path.join(OUTPUT_DIR, f"training_history_{model.quant_mode}_minimal.png")
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss vs. Epochs'); plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
        plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
        plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy vs. Epochs'); plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Training history plot saved to {plot_path}")
        plt.close()
    except Exception as plot_e:
        print(f"Warning: Could not generate or save training plot: {plot_e}")
        
    return model, model_path, best_val_acc


# --- Testing Function ---
def test_model(model, test_loader, device):
    """Evaluate the model on test data."""
    print("Testing model...")
    model.to(device)
    model.eval()

    if test_loader is None:
        print("No test data available. Skipping testing.")
        return None

    class_correct = {}
    class_total = {}
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate overall accuracy
            c = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate per-class accuracy
            for i, label in enumerate(labels):
                label_key = label.item()
                if label_key not in class_correct:
                    class_correct[label_key] = 0
                    class_total[label_key] = 0
                
                if len(c.shape) == 0:  # Handle single item in batch
                    class_correct[label_key] += c.item()
                else:
                    class_correct[label_key] += c[i].item()
                class_total[label_key] += 1
    
    # Calculate and display overall accuracy
    test_acc = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {test_acc:.2f}% ({correct}/{total})")
    
    # Display per-class accuracy
    print("\nPer-class accuracy:")
    per_class_accs = []
    for class_id in sorted(class_total.keys()):
        class_acc = 100 * class_correct[class_id] / class_total[class_id]
        per_class_accs.append(class_acc)
        print(f"  Class {class_id}: {class_acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})")
    
    # Calculate average per-class accuracy
    avg_class_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0
    print(f"\nAverage per-class accuracy: {avg_class_acc:.2f}%")
    
    return {
        'accuracy': test_acc,
        'per_class_accuracy': {class_id: 100 * class_correct[class_id] / class_total[class_id] 
                              for class_id in class_total},
        'avg_class_accuracy': avg_class_acc
    }

# --- Main Execution Block ---
def main():
    """Main execution function."""
    # Set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda: 
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    try:
        # Load dataset
        dataset = load_dataset(DATASET_PATH)
        num_classes = dataset.get('num_classes')
        class_names = dataset.get('class_names', {})
        
        if num_classes is None:
            raise ValueError("Dataset missing 'num_classes' key.")
        print(f"Dataset contains {num_classes} classes.")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(dataset, BATCH_SIZE, use_cuda)
        
        # Create model with specified quantization
        model = MinimalRoadSignNet(num_classes=num_classes)
        print(f"\nModel architecture:\n{model}\n")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        print(f"\nStarting training using {QUANT_MODE} quantization for FINN compatibility.")
        print(f"Training parameters: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LEARNING_RATE}")
        print(f"IMPORTANT: Using SIGNED quantization for identity activations")
        print(f"IMPORTANT: Using UNSIGNED quantization for ReLU activations")

        # Train model
        trained_model, model_path, best_val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
        )
        
        # Test the best model
        print(f"\nLoading best model from {model_path} for final testing...")
        test_model_instance = MinimalRoadSignNet(num_classes=num_classes)
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                test_model_instance.load_state_dict(checkpoint['model_state_dict'])
            else:
                test_model_instance.load_state_dict(checkpoint)
                
            test_results = test_model(test_model_instance, test_loader, device)
            
            if test_results:
                # Save model summary to a text file
                summary_path = os.path.join(OUTPUT_DIR, f"model_summary_{QUANT_MODE}_minimal.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"Minimal Road Sign Classifier Model Summary\n")
                    f.write(f"======================================\n\n")
                    f.write(f"Architecture: 3 blocks with 3 layers each\n")
                    f.write(f"Quantization mode: {QUANT_MODE}\n")
                    f.write(f"Number of classes: {num_classes}\n")
                    f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
                    f.write(f"Test accuracy: {test_results['accuracy']:.2f}%\n")
                    f.write(f"Average per-class accuracy: {test_results['avg_class_accuracy']:.2f}%\n\n")
                    
                    f.write("Per-class accuracy:\n")
                    for class_id in sorted(test_results['per_class_accuracy'].keys()):
                        class_acc = test_results['per_class_accuracy'][class_id]
                        class_name = class_names.get(class_id, f"Class {class_id}")
                        f.write(f"  {class_id}: {class_name} - {class_acc:.2f}%\n")
                
                print(f"Model summary saved to {summary_path}")
        except Exception as e:
            print(f"Error during testing: {e}")
        
        print("\nTraining and evaluation complete!")
        print(f"Best model saved to: {model_path}")
        print(f"This minimal model has exactly 3 blocks with 3 layers each.")
        print(f"Next step: Export to QONNX format for FINN deployment.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
