import gradio as gr
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os

model = torch.jit.load("EdgeCNN_mobile.pt", map_location="cpu")
model.eval()

emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ✅ Preprocessing function (with debug)
def preprocess(img, debug=False):
    img = img.convert("L")  # greyscale
    img = np.array(img)

    if np.mean(img) > 127:  # invert if light background
        img = 255 - img

    coords = np.column_stack(np.where(img < 255))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1+1, x0:x1+1]

    h, w = img.shape
    pad = max(h, w)
    pad_h = (pad - h) // 2
    pad_w = (pad - w) // 2
    img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=255)

    img = np.rot90(img, k=3)  # rotate to match model input
    img = np.fliplr(img)
    
    # convert to PIL and enhance the pil image
    img_pil = Image.fromarray(img).resize((28, 28))
    img_pil = img_pil.filter(ImageFilter.SHARPEN)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(2.5) # recommand: 1.5~3 
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.5)

    # ✅ Save debug image if needed
    if debug:
        img_pil.save("debug_processed.png")  # you can check this file locally

    return img_pil


# ✅ Enhanced predict that returns debug image too
def predict(image):
    img = preprocess(image, debug=False)  # don't save file, just return
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        top3 = torch.topk(probs, 3)
        pred = {emnist_classes[i]: float(probs[i]) for i in top3.indices}
        return pred, img  # return prediction + image



# ✅ Model information
model_info = f"""
### 📦 Mobile Model Info

- **Model Type**: TorchScript traced Mobile EdgeCNN  
- **Parameters**: 14,928  
- **Model Size**: 0.24 MB  
- **Accuracy**: 83.8%  
- **Avg Inference Latency**: 1.37 ms  
- **P95 Latency**: 1.69 ms  
- **Memory Usage**: 0.22 MB  
- **Throughput**: 731.8 FPS  
- **Compatibility**: ✅ iPhone 14, Galaxy S23, Pi 4B, Jetson Nano  
- **Production Grade**: 🥇 Ready for Deployment
"""

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 EMNIST Edge Classifier 1 (Mobile Version)")
    gr.Markdown(model_info)
    
    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Handwritten Character")
        output_label = gr.Label(num_top_classes=3, label="Top 3 Predictions")
        processed_img = gr.Image(label="Processed Image")

    input_img.change(fn=predict, inputs=input_img, outputs=[output_label, processed_img])

    # with gr.Row():
    #     input_img = gr.Image(type="pil", label="Upload Handwritten Character")
    #     output_label = gr.Label(num_top_classes=3, label="Top 3 Predictions")
    
    # input_img.change(fn=predict, inputs=input_img, outputs=output_label)



# ✅ Launch the Gradio app
if __name__ == "__main__":
    # local development
    # demo.launch()
    
    # production deployment
    port = int(os.environ.get("PORT", 8000))
    demo.launch(server_name="0.0.0.0", server_port=port)