import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import uniform
from sklearn.model_selection import ParameterSampler
from skopt import gp_minimize
from skopt.space import Categorical, Real

from data import get_data_loaders
from model import Model
from train import build_loss_fn, build_optimizer, train_model


RESULTS_DIR = Path(__file__).parent / "results"
GREEK_DIR = Path(__file__).resolve().parent / "Greek"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search vs Bayesian optimization for the CNN")
    parser.add_argument("--random-iters", type=int, default=20)
    parser.add_argument("--bo-iters", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=105)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1422)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_hyperparams(
    learning_rate: float,
    dropout: float,
    batch_size: int,
    optimizer_name: str,
    criterion_name: str,
    args: argparse.Namespace,
    device: str,
) -> float:
    train_loader, val_loader = get_data_loaders(
        GREEK_DIR,
        batch_size=batch_size,
        train_split=args.train_split,
        img_size=args.img_size,
        seed=args.seed,
    )

    model = Model(dropout=dropout, num_classes=24)
    optimizer = build_optimizer(model, optimizer_name=optimizer_name, learning_rate=learning_rate)
    loss_fn = build_loss_fn(criterion_name)

    return train_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        device=device,
    )


def to_serializable_search_point(point: list[object]) -> dict[str, object]:
    return {
        "learning_rate": float(point[0]),
        "dropout": float(point[1]),
        "batch_size": int(point[2]),
        "optimizer": str(point[3]),
        "criterion": str(point[4]),
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    device = resolve_device(args.device)

    domain_random = {
        "learning_rate": uniform(0.0005, 0.0495),
        "dropout": uniform(0.05, 0.45),
        "batch_size": [16, 32, 64],
        "optimizer": ["adam", "sgd"],
        "criterion": ["crossentropy", "multimargin"],
    }

    print("Random search candidates")
    param_list = list(ParameterSampler(domain_random, n_iter=args.random_iters, random_state=args.seed))
    print(param_list)

    current_best = 0.0
    max_acc_per_iter = []
    random_history = []

    for index, params in enumerate(param_list, start=1):
        print(f"\nRandom search iteration {index}/{args.random_iters}")
        print(params)
        start = time.time()

        val_acc = evaluate_hyperparams(
            learning_rate=float(params["learning_rate"]),
            dropout=float(params["dropout"]),
            batch_size=int(params["batch_size"]),
            optimizer_name=str(params["optimizer"]),
            criterion_name=str(params["criterion"]),
            args=args,
            device=device,
        )

        current_best = max(current_best, float(val_acc))
        max_acc_per_iter.append(current_best)
        random_history.append(
            {
                "params": {
                    "learning_rate": float(params["learning_rate"]),
                    "dropout": float(params["dropout"]),
                    "batch_size": int(params["batch_size"]),
                    "optimizer": str(params["optimizer"]),
                    "criterion": str(params["criterion"]),
                },
                "val_accuracy": float(val_acc),
                "elapsed_seconds": time.time() - start,
            }
        )
        print(f"Validation accuracy: {val_acc:.4f}")

    x0 = [
        float(param_list[0]["learning_rate"]),
        float(param_list[0]["dropout"]),
        int(param_list[0]["batch_size"]),
        str(param_list[0]["optimizer"]),
        str(param_list[0]["criterion"]),
    ]
    y0 = -float(random_history[0]["val_accuracy"])

    search_space = [
        Real(0.0005, 0.05, name="learning_rate"),
        Real(0.05, 0.5, name="dropout"),
        Categorical([16, 32, 64], name="batch_size"),
        Categorical(["adam", "sgd"], name="optimizer"),
        Categorical(["crossentropy", "multimargin"], name="criterion"),
    ]

    objective_calls = []

    def objective_function(x: list[object]) -> float:
        start = time.time()
        val_acc = evaluate_hyperparams(
            learning_rate=float(x[0]),
            dropout=float(x[1]),
            batch_size=int(x[2]),
            optimizer_name=str(x[3]),
            criterion_name=str(x[4]),
            args=args,
            device=device,
        )
        objective_calls.append(
            {
                "params": to_serializable_search_point(x),
                "val_accuracy": float(val_acc),
                "elapsed_seconds": time.time() - start,
            }
        )
        print(f"BO iteration {len(objective_calls)}/{args.bo_iters}: val_acc={val_acc:.4f}")
        return -float(val_acc)

    bo_additional_calls = max(args.bo_iters - 1, 0)
    opt = gp_minimize(
        objective_function,
        search_space,
        acq_func="EI",
        n_initial_points=0,
        n_calls=bo_additional_calls,
        x0=[x0],
        y0=[y0],
        xi=0.1,
        noise=0.01**2,
        random_state=args.seed,
    )

    bo_func_vals = np.asarray(opt.func_vals, dtype=float)
    bo_best_per_iter = np.maximum.accumulate(-bo_func_vals).ravel()
    random_best_per_iter = np.asarray(max_acc_per_iter, dtype=float)

    print("\nOptimization finished")
    print("Best encoded x:", opt.x)
    print("Best objective (neg val acc):", opt.fun)
    print("Best hyperparameters:", to_serializable_search_point(opt.x))

    plot_path = RESULTS_DIR / "bo_vs_random_search.png"
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(random_best_per_iter) + 1), random_best_per_iter, "o-", color="tab:red", label="Random Search")
    plt.plot(range(1, len(bo_best_per_iter) + 1), bo_best_per_iter, "o-", color="tab:blue", label="Bayesian Optimization")
    plt.xlabel("Iterations")
    plt.ylabel("Best validation accuracy")
    plt.title("Random Search vs Bayesian Optimization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "img_size": args.img_size,
            "train_split": args.train_split,
            "device": device,
            "random_iters": args.random_iters,
            "bo_iters": args.bo_iters,
        },
        "random_search": random_history,
        "bayesian_optimization": {
            "evaluations": objective_calls,
            "best_x": to_serializable_search_point(opt.x),
            "best_val_accuracy": float(-opt.fun),
            "func_vals": bo_func_vals.tolist(),
        },
        "max_acc_per_iter": random_best_per_iter.tolist(),
        "bo_func_vals": bo_func_vals.tolist(),
    }

    results_path = RESULTS_DIR / "bo_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Results saved to: {results_path}")
    print(f"Plot saved to:    {plot_path}")


if __name__ == "__main__":
    main()
