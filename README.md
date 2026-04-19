# Road Sign Classification on PYNQ-Z2 with FINN

**Real-time road sign detection using quantized neural networks on FPGA**

[![Platform](https://img.shields.io/badge/Platform-PYNQ--Z2-orange.svg)](https://www.pynq.io/)
[![FINN](https://img.shields.io/badge/FINN-W4A4-blue.svg)](https://github.com/Xilinx/finn)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)

---

## 📌 Overview

This project implements real-time road sign classification on the PYNQ-Z2 FPGA board using FINN (Fast, Scalable Quantized Neural Network Inference). The neural network uses W4A4 quantization (4-bit weights, 4-bit activations) for efficient hardware acceleration.

**Complete Pipeline:**
```
Dataset Prep → Training → ONNX Export → FINN Compilation → FPGA Bitstream → Real-time Inference
```

---

## 🗂️ Project Structure

```
SAFE_PynqZ2/
├── src/
│   ├── extract_class_labels.py              # Step 1: Extract dataset labels
│   ├── prepare_dataset.py                   # Step 2: Prepare dataset
│   ├── train.py                             # Step 3: Train model → .pt file
│   └── video_inference_new.py               # Step 7: Real-time inference on PYNQ
│
├── docker_ipynb/
│   ├── road_sign_classifier_executable.ipynb    # Step 4: PyTorch → ONNX
│   └── FINNRoadSignNet_w4a4_finnCompile.ipynb   # Step 5: ONNX → Bitstream
│
├── finn_outputs/
│   ├── road_sign_minimal_finn_w4a4.onnx         # Quantized ONNX model
│   ├── folding_config_minimal.json              # FINN folding config
│   ├── pynq_inference.ipynb                     # Step 6: PYNQ test notebook
│   ├── FINNRoadSignNet_w4a4_finnCompile.bit     # FPGA bitstream
│   └── FINNRoadSignNet_w4a4_finnCompile.hwh     # Hardware handoff
│
└── README.md
```

---

## 🚀 Complete Workflow (7 Steps)

### Prerequisites

| Environment | Requirements |
|-------------|-------------|
| **Local Machine** | Python 3.8+, PyTorch, ONNX |
| **FINN Docker** | Docker, FINN repository, Jupyter |
| **PYNQ-Z2** | PYNQ image v2.7+, USB webcam |

---

### STEP 1 & 2: Dataset Preparation

Run on your **local machine**:

```bash
# Extract class labels from dataset
python src/extract_class_labels.py

# Prepare and preprocess the dataset
python src/prepare_dataset.py
```

**Output:** Organized dataset ready for training.

**Expected dataset structure:**
```
dataset/
├── stop_sign/
├── speed_limit/
├── yield/
└── ...
```

---

### STEP 3: Model Training

Run on your **local machine**:

```bash
python src/train.py
```

**Output:** `road_sign_model.pt` (PyTorch model checkpoint)

Configure training in `train.py`:
```python
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

---

### STEP 4: ONNX Conversion (FINN Docker)

⚠️ **Must run inside the FINN Docker environment.**

```bash
# Clone FINN repository
git clone https://github.com/Xilinx/finn.git
cd finn

# Launch FINN Docker with Jupyter
./run-docker.sh notebook

# Open browser at http://localhost:8888
# Navigate to and run: docker_ipynb/road_sign_classifier_executable.ipynb
```

**What it does:** Loads PyTorch `.pt` file, applies W4A4 quantization, exports ONNX.

**Output:** `road_sign_minimal_finn_w4a4.onnx`

---

### STEP 5: FINN Compilation to Bitstream

⚠️ **Continue in the FINN Docker environment.**

In Jupyter, run: `docker_ipynb/FINNRoadSignNet_w4a4_finnCompile.ipynb`

**What it does:**
- Takes the W4A4 quantized ONNX model
- Applies FINN streamlining and transformations
- Synthesizes hardware accelerator for PYNQ-Z2
- Generates FPGA bitstream via Vivado

**Output:**
- `FINNRoadSignNet_w4a4_finnCompile.bit` — FPGA bitstream
- `FINNRoadSignNet_w4a4_finnCompile.hwh` — Hardware handoff file

Adjust parallelism in `folding_config_minimal.json` to tune latency vs resources.

---

### STEP 6: Deploy to PYNQ-Z2

Transfer the generated files and set up the PYNQ board:

```bash
# Copy bitstream files to PYNQ-Z2
scp finn_outputs/*.bit finn_outputs/*.hwh xilinx@pynq:~/road_sign_detection/

# SSH into PYNQ-Z2
ssh xilinx@pynq

# Launch Jupyter on PYNQ
cd ~/road_sign_detection
jupyter notebook --ip=0.0.0.0
```

Open `pynq_inference.ipynb` in the browser at `http://<pynq-ip>:8888` and run all cells to validate the accelerator.

---

### STEP 7: Real-time Video Inference

Connect a USB webcam to PYNQ-Z2 and run:

```bash
python src/video_inference_new.py
```

**What happens:**
- FPGA bitstream loads onto programmable logic
- Webcam initializes and streams frames
- Each frame is classified in real-time on FPGA
- Display shows live video with predicted road sign labels
- Press `q` to quit

---

## 🎓 Technical Details

### Model
- **Input:** 32×32 RGB images
- **Network:** Custom CNN optimized for FINN
- **Quantization:** W4A4 (4-bit weights, 4-bit activations)
- **Output:** Multi-class road sign classification

### FINN Compilation Flow
1. Streamlining and ONNX optimization
2. Quantization-aware transformations
3. Folding configuration for PYNQ-Z2 resources
4. Hardware generation (Verilog)
5. Vivado synthesis and place & route
6. Bitstream generation (`.bit` and `.hwh`)

### Performance (approximate)
- **Inference speed:** 30+ FPS on PYNQ-Z2
- **Latency:** <10 ms per frame
- **Power:** ~3 W total system

---

## 🐛 Troubleshooting

**FINN Docker won't start:**
```bash
docker ps                 # Check Docker is running
cd finn && git pull       # Update FINN
./run-docker.sh notebook
```

**ONNX export fails:**
- Set `model.eval()` before export
- Verify input tensor shape
- Ensure all layers are ONNX-compatible

**Bitstream won't load on PYNQ:**
```python
import os, pynq
print(os.path.exists('bitstream.bit'))
print(pynq.__version__)  # Should be 2.7+
```

**Low inference FPS:**
- Reduce webcam resolution
- Increase folding parallelism and recompile
- Verify bitstream actually loaded

---

## 📚 Resources

- [FINN GitHub](https://github.com/Xilinx/finn)
- [FINN Documentation](https://finn.readthedocs.io/)
- [FINN Examples](https://github.com/Xilinx/finn-examples)
- [PYNQ Documentation](http://pynq.readthedocs.io/)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Xilinx FINN team — for the FINN framework
- PYNQ project — for the development platform
- PyTorch community — for deep learning tools

---

**Built with FINN on PYNQ-Z2 🚦**
