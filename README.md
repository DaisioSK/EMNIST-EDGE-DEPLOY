# 🧠 EMNIST Edge Classifier (Mobile Version)

A compact and efficient handwritten character recognizer, optimized for edge devices and deployed with CI/CD to Azure App Service.

---

## 🚀 Project Highlights

- 📦 **Tiny & Efficient**: Just **14,928 parameters**, **0.24MB** TorchScript model.
- ⚡ **Fast Inference**: Average latency of **1.37ms**, **731.8 FPS** throughput.
- 🧠 **Solid Accuracy**: 83.8% top-1 accuracy on EMNIST-balanced test set.
- 🌐 **CI/CD Integrated**: Automatic deployment pipeline from GitHub to Azure.
- 📱 **Edge Ready**: Tested on iPhone 14, Galaxy S23, Raspberry Pi 4B, Jetson Nano.
- 🧪 **Gradio Frontend**: Clean UI with top-3 predictions and static model info.

---

## 🗂️ Project Structure

```
EMNIST-EDGE/
├── app.py                   # Gradio app
├── EdgeCNN_mobile.pt        # TorchScript model
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image for Azure
├── .github/workflows/       # CI/CD via GitHub Actions
├── notebooks/               # Model training and profiling (Colab)
└── images/, test_samples/   # Demo and testing materials
```

---

## 🧪 Local Usage

```bash
conda create -n emnist-edge python=3.10
conda activate emnist-edge

pip install -r requirements.txt
python app.py
```

---

## 🔁 CI/CD + Azure Deployment

This project integrates with **GitHub Actions** to enable push-triggered deployment via Docker image to **Azure App Service for Containers**.

### ✅ Summary

- CI/CD pipeline: GitHub Actions → ACR → Azure Web App
- Auto-builds and pushes Docker image on code update
- Model is loaded via `app.py` and served with Gradio
- Live at:
  [🌐 Click to Open App](https://emnidt-edge-container-app-gkevhscfbchub4eq.southeastasia-01.azurewebsites.net/)

---

## 📊 Model Metadata

| Metric              | Value        |
|---------------------|--------------|
| Parameters          | 14,928       |
| Size (TorchScript)  | 0.24 MB      |
| Accuracy            | 83.8%        |
| Inference Time      | 1.37 ms      |
| P95 Latency         | 1.69 ms      |
| Throughput          | 731.8 FPS    |
| Memory Footprint    | 0.22 MB      |

---

## 📓 Notebooks (Optional)

- `EMNIST.ipynb`: Baseline CNN training
- `EMNIST_Edge.ipynb`: Edge model with quantization + profiling + TorchScript export

---

## 📌 Acknowledgements

- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Gradio UI Framework](https://gradio.app)
- Azure App Service & GitHub Actions for deployment support

---

## 🛠️ Future Plans

- ONNX + TFLite export for broader edge support  
- Client-side latency reporting module on edge device