import torch
from pathlib import Path
from model import Model
from data import get_data_loaders
from train import train_model


def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get data loaders
    greek_dir = Path(__file__).parent / 'Greek'
    print(f"Loading data from: {greek_dir}")
    
    train_loader, val_loader = get_data_loaders(
        greek_dir,
        batch_size=32,
        train_split=0.8,
        img_size=64
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = Model(in_channels=3, num_classes=24)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    print("\nStarting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=10,
        learning_rate=0.001,
        device=device
    )


if __name__ == "__main__":
    main()
