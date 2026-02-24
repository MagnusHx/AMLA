import torch
from pathlib import Path
from model import Model
from data import get_data_loaders


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
        img_size=105
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    

    model = Model(in_channels=1, num_classes=24)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train using model.fit
    print("\nStarting training...")
    model = model.fit(
        train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.001,
        device=device
    )

    # Final evaluation on validation set and return metrics
    val_loss, val_acc = model.evaluate(val_loader, device=device)
    print(f"\nFinal validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
    return val_loss, val_acc


if __name__ == "__main__":
    metrics = main()
    # metrics is (val_loss, val_acc)
    print(f"Returned metrics: {metrics}")
