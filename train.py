"""
Training script for flower classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time

import config
from data_loader import get_data_loaders
from classifier import create_model
from helper import save_checkpoint, plot_loss_accuracy

def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
    
    Returns:
        val_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / total
    
    return val_loss, accuracy

def train_model():
    """Train the flower classification model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, class_names = get_data_loaders()
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create model
    model = create_model(len(class_names))
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate epoch loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Validate the model
        epoch_val_loss, accuracy = validate(model, val_loader, criterion, device)
        
        # Store losses and accuracy
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        accuracies.append(accuracy)
        
        # Print epoch results
        end_time = time.time()
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
              f"Time: {end_time - start_time:.2f}s - "
              f"Train Loss: {epoch_train_loss:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f} - "
              f"Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(
                model, optimizer, epoch+1, epoch_train_loss, 
                epoch_val_loss, accuracy, config.MODEL_SAVE_PATH
            )
    
    # Plot training curves
    plot_loss_accuracy(train_losses, val_losses, accuracies)
    
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    return model, class_names

if __name__ == "__main__":
    train_model()