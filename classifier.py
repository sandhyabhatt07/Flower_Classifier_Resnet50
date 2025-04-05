"""
Model definition for flower classification using transfer learning
"""

import torch
import torch.nn as nn
from torchvision import models
import config

def create_model(num_classes=None):
    """
    Create a pre-trained ResNet50 model with modified final layer
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        A PyTorch model
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    # Load pre-trained ResNet model
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model

def load_checkpoint(model, checkpoint_path):
    """
    Load model checkpoint from file
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model