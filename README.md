# 🌸 Flower Classification Web App

A deep learning-based web application for classifying flower images into one of 14 categories using **ResNet50** and **Flask**. It supports both image uploads and clickable sample images for quick prediction.

---

## 🚀 Features

- 🔍 Upload an image or click on a sample image to classify it
- ⚙️ Transfer Learning using ResNet50 (pretrained on ImageNet)
- 📊 Shows the predicted flower class and confidence score
- 🖼 Displays the uploaded/clicked image with the prediction
- 🌐 Clean and interactive web interface using Flask

---

## 🧠 Model

- **Architecture**: ResNet50 (pretrained)
- **Training**: Fine-tuned on custom flower dataset with 14 classes
- **Framework**: PyTorch
- **Inference**: CPU-friendly, quick response time

---

## 🛠 Tech Stack

- Python, Flask
- PyTorch, TorchVision
- HTML, CSS (Jinja templating)
- Pillow, Pandas, TQDM
- Matplotlib (for training visualization)

---

**📸 Data Augmentation**
Applied using torchvision.transforms, including:

Random Horizontal Flip

Random Rotation

Resize & Normalization

## ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/sandhyabhatt07/Flower_Classification.git
cd Flower_Classification


python -m venv venv
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On macOS/Linux


pip install -r requirements.txt

python app.py



