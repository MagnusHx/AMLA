import argparse
from pathlib import Path

import torch

from data import get_data_loaders
from model import Model
from train import build_loss_fn, build_optimizer, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN on the Greek character dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=105)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument(
        "--criterion",
        type=str,
        default="crossentropy",
        choices=["crossentropy", "multimargin"],
    )
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1422)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    greek_dir = Path(__file__).parent / "Greek"

    print(f"Using device: {device}")
    print(f"Loading data from: {greek_dir}")

    train_loader, val_loader = get_data_loaders(
        greek_dir,
        batch_size=args.batch_size,
        train_split=args.train_split,
        img_size=args.img_size,
        seed=args.seed,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = Model(in_channels=1, num_classes=24, dropout=args.dropout)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = build_optimizer(model, optimizer_name=args.optimizer, learning_rate=args.learning_rate)
    loss_fn = build_loss_fn(args.criterion)

    print("\nStarting training...")
    best_val_accuracy = train_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        device=device,
    )

    print(f"\nBest validation accuracy (fraction): {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
