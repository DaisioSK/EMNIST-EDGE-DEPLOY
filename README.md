# 🧠 EMNIST Edge Classifier (Mobile Version)

A lightweight, edge-optimized handwritten character recognizer built with PyTorch, trained on the EMNIST dataset, and deployed via Gradio.

---

## 🚀 Project Highlights

- 📦 **Mobile-Ready Deployment**: Exported as a TorchScript model with only **14,928 parameters** and a **0.24MB** file size.
- ⚡ **Fast & Efficient**: Achieves **1.37 ms** average inference latency with **731.8 FPS throughput**.
- 🧠 **Model Accuracy**: 83.8% top-1 accuracy on EMNIST-balanced test set.
- 📱 **Compatibility**: Verified on iPhone 14, Galaxy S23, Raspberry Pi 4B, Jetson Nano.
- 🎨 **Gradio Frontend**: Interactive UI with top-3 prediction visualization and static model info panel.

---

## 📁 Project Structure

```bash
EMNIST-EDGE/
├── images/                  # Example prediction images
├── notebooks/               # Training and optimization notebooks
│   ├── EMNIST.ipynb         # Original CNN training (Colab)
│   └── EMNIST_Edge.ipynb    # Mobile version training + profiling
├── test_samples/            # Manual test images
├── app.py                   # Gradio inference app
├── EdgeCNN_mobile.pt        # TorchScript traced mobile model
├── requirements.txt         # Python dependencies
└── .gitignore               # Ignore cache and large logs
```

---

## 🎯 Usage

```bash
# Step 1: Create conda env (optional)
conda create -n emnist-edge python=3.10
conda activate emnist-edge

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run app
python app.py
```

---

## 📓 Notebooks (Optional)

Training and evaluation workflows are available in the [`notebooks/`](notebooks/) folder, including:

- `EMNIST.ipynb`: Standard CNN for full-size EMNIST
- `EMNIST_Edge.ipynb`: Optimized MobileCNN with quantization, pruning, profiling, and TorchScript tracing

---

## 🔍 Model Metadata (EdgeCNN_mobile.pt)

| Metric              | Value         |
|---------------------|---------------|
| Parameters          | 14,928        |
| File Size           | 0.24 MB       |
| Accuracy            | 83.8%         |
| Avg Inference Time  | 1.37 ms       |
| P95 Latency         | 1.69 ms       |
| Throughput          | 731.8 FPS     |
| Memory Usage        | 0.22 MB       |
| Production Grade    | ✅ Ready       |

---

## 📌 Acknowledgements

- Dataset: [EMNIST by Cohen et al.](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- Interface: [Gradio](https://gradio.app)
- Edge Deployment: TorchScript tracing + profiling on Colab

---

## 🧪 Future Work

- 📦 Dockerized deployment on Azure App Service
- 📱 Convert to ONNX and TFLite
- 📉 Real-time latency benchmarking from client device