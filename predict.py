"""
Script for making predictions on a random image from a folder
"""

import argparse
import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

from classifier import create_model, load_checkpoint
from helper import predict_image
import config

def display_image(image_path, predicted_class, confidence, true_class=None):
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.axis('off')

    if true_class:
        title = f"Predicted: {predicted_class} ({confidence*100:.2f}%)\nTrue: {true_class}"
    else:
        title = f"Prediction: {predicted_class} ({confidence*100:.2f}%)"

    plt.title(title)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict flower class from a random image in a folder')
    parser.add_argument('--folder', type=str, required=True, help='Path to folder with images')
    parser.add_argument('--model', type=str, default=config.MODEL_SAVE_PATH, help='Path to model checkpoint')
    args = parser.parse_args()

    # Get all image files
    image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.folder)
        for file in files
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print("No image files found in the folder.")
        return

    # Select a random image and extract true class from folder
    random_image = random.choice(image_files)
    true_class_name = os.path.basename(os.path.dirname(random_image))
    print(f"\nRandomly selected image: {random_image}")
    print(f"True class (from folder): {true_class_name}")

    # Load class names from training directory (sorted alphabetically)
    class_names = sorted(entry.name for entry in os.scandir(config.TRAIN_DIR) if entry.is_dir())

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(class_names))
    model = load_checkpoint(model, args.model)
    model = model.to(device)

    # Predict
    predicted_class_name, confidence = predict_image(model, random_image, class_names)

    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {confidence:.4f}")
    print(f"(Debug) Predicted class name: {predicted_class_name}")

    # Compare prediction with true class
    if predicted_class_name == true_class_name:
        print("✅ Prediction matches the true class!")
    else:
        print("❌ Prediction does not match the true class.")

    # Display image with prediction and true class
    display_image(random_image, predicted_class_name, confidence, true_class=true_class_name)

if __name__ == "__main__":
    main()
