import torch
import torch.nn as nn
from tqdm import tqdm


def build_optimizer(model, optimizer_name='adam', learning_rate=0.001):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_loss_fn(loss_name='crossentropy'):
    if loss_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    if loss_name == 'multimargin':
        return nn.MultiMarginLoss()
    raise ValueError(f"Unsupported loss function: {loss_name}")


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, data_loader, loss_fn=None, device='cpu'):
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    return validate(model, data_loader, loss_fn, device)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    loss_fn=None,
    num_epochs=10,
    device='cpu',
    learning_rate=0.001,
    optimizer_name='adam',
):
    """
    Training loop for the Greek character recognition model.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Configured optimizer (e.g. Adam, SGD). If omitted, one is created.
        loss_fn: Loss function (e.g. CrossEntropyLoss, MultiMarginLoss). If omitted, CrossEntropy is used.
        num_epochs: Number of epochs to train
        device: Device to train on ('cpu' or 'cuda')
        learning_rate: Learning rate used when optimizer is created internally
        optimizer_name: Optimizer name used when optimizer is created internally

    Returns:
        best_val_accuracy: Best validation accuracy achieved (as a fraction, 0-1)
    """
    model.to(device)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = optimizer or build_optimizer(model, optimizer_name=optimizer_name, learning_rate=learning_rate)

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved with validation accuracy: {val_acc:.2f}%")

    print(f"\nTraining complete! Best validation accuracy: {best_val_accuracy:.2f}%")
    return best_val_accuracy / 100  # return as fraction for consistency with baysian.py
