"""
Helper functions for the flower classification project
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import config

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, accuracy, filepath):
    """
    Save model checkpoint to file
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        accuracy: Validation accuracy
        filepath: Path to save checkpoint file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'accuracy': accuracy
    }, filepath)
    
    print(f"Checkpoint saved to {filepath}")

def plot_loss_accuracy(train_losses, val_losses, accuracies):
    """
    Plot training/validation loss and accuracy curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        accuracies: List of validation accuracies per epoch
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def predict_image(model, image_path, class_names):
    """
    Make prediction on a single image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to image file
        class_names: List of class names
        
    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted_class_idx.item()]
    confidence = confidence.item()
    
    return predicted_class, confidence