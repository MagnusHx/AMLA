import argparse
import json
import os
import random
import warnings
from collections import namedtuple
from pathlib import Path

MPLCONFIGDIR = Path(__file__).parent / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import Model


warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 1422
IMG_SIZE = 105
TEST_SIZE = 0.2
VAL_SIZE = 0.25
N_QUERIES = 20
N_REPEATS = 1
COMMITTEE_SIZE = 3
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DROPOUT = 0.2
SAMPLES_PER_CLASS = 1
NUM_CLASSES = 24
RESULTS_DIR = Path(__file__).parent / "results"
GREEK_DIR = Path(__file__).parent / "Greek"

ResultsRecord = namedtuple("ResultsRecord", ["estimator", "query_id", "score"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_greek_dataset(root_dir: Path, img_size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    for class_idx, class_dir in enumerate(sorted(root_dir.glob("character*"))):
        for image_path in sorted(class_dir.glob("*.png")):
            image = Image.open(image_path).convert("L").resize((img_size, img_size))
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_array = (image_array - 0.5) / 0.5
            images.append(image_array[np.newaxis, :, :])
            labels.append(class_idx)

    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class TorchCNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        learning_rate: float = LEARNING_RATE,
        dropout: float = DROPOUT,
        batch_size: int = BATCH_SIZE,
        max_epochs: int = EPOCHS,
        device: str | None = None,
        random_state: int = SEED,
        reset_on_fit: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.random_state = random_state
        self.reset_on_fit = reset_on_fit

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_model(self) -> None:
        torch.manual_seed(self.random_state)
        self.device_ = self._resolve_device()
        self.model_ = Model(dropout=self.dropout, num_classes=self.num_classes).to(self.device_)
        self.loss_fn_ = nn.CrossEntropyLoss()
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        self.classes_ = np.arange(self.num_classes)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]

        if not hasattr(self, "model_") or self.reset_on_fit:
            self._build_model()

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        self.model_.train()
        for _ in range(self.max_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                logits = self.model_(batch_x)
                loss = self.loss_fn_(logits, batch_y)
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]

        tensor_x = torch.from_numpy(X).to(self.device_)
        self.model_.eval()

        probabilities = []
        with torch.no_grad():
            for start in range(0, len(tensor_x), self.batch_size):
                batch_x = tensor_x[start : start + self.batch_size]
                logits = self.model_(batch_x)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

        return np.concatenate(probabilities, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))


def build_model(seed: int, args: argparse.Namespace) -> TorchCNNClassifier:
    return TorchCNNClassifier(
        num_classes=NUM_CLASSES,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        device=args.device,
        random_state=seed,
    )


def aggregate_records(results: list[ResultsRecord]) -> dict[str, list[float]]:
    grouped_scores: dict[int, list[float]] = {}

    for record in results:
        grouped_scores.setdefault(record.query_id, []).append(float(record.score))

    query_ids = sorted(grouped_scores)
    mean_accuracy = [float(np.mean(grouped_scores[query_id])) for query_id in query_ids]
    std_accuracy = [float(np.std(grouped_scores[query_id])) for query_id in query_ids]

    return {
        "query_ids": query_ids,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
    }


def choose_initial_indices(y_train: np.ndarray, samples_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected_indices = []

    for class_id in np.unique(y_train):
        class_indices = np.flatnonzero(y_train == class_id)
        chosen = rng.choice(
            class_indices,
            size=min(samples_per_class, len(class_indices)),
            replace=False,
        )
        selected_indices.extend(chosen.tolist())

    return np.asarray(sorted(selected_indices), dtype=int)


def make_pool_indices(n_samples: int, initial_indices: np.ndarray) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    mask[initial_indices] = False
    return np.flatnonzero(mask)


def run_random_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[ResultsRecord], list[float]]:
    random_results = []
    final_test_scores = []

    for i_repeat in range(args.n_repeats):
        initial_indices = choose_initial_indices(
            y_train,
            samples_per_class=args.samples_per_class,
            seed=args.seed + i_repeat,
        )
        pool_indices = make_pool_indices(len(x_train), initial_indices)
        rng = np.random.default_rng(args.seed + i_repeat)
        rng.shuffle(pool_indices)

        learner = build_model(args.seed + i_repeat, args)
        labeled_indices = initial_indices.copy()
        learner.fit(x_train[labeled_indices], y_train[labeled_indices])

        max_queries = min(args.n_queries, len(pool_indices))
        for i_query in tqdm(
            range(1, max_queries + 1),
            desc=f"Random CNN repeat {i_repeat + 1}/{args.n_repeats}",
            leave=False,
        ):
            new_index = pool_indices[i_query - 1]
            labeled_indices = np.append(labeled_indices, new_index)
            learner.fit(x_train[labeled_indices], y_train[labeled_indices])
            score = learner.score(x_val, y_val)
            random_results.append(ResultsRecord("random_cnn", i_query, score))

        final_test_scores.append(float(learner.score(x_test, y_test)))

    return random_results, final_test_scores


def run_qbc_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[ResultsRecord], list[int], list[float]]:
    try:
        from modAL.disagreement import vote_entropy_sampling
        from modAL.models import ActiveLearner, Committee
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "modAL is required for this script because the exercise uses "
            "ActiveLearner, Committee, and vote_entropy_sampling."
        ) from exc

    committee_results = []
    first_run_queried_labels = []
    final_test_scores = []

    for i_repeat in range(args.n_repeats):
        initial_indices = choose_initial_indices(
            y_train,
            samples_per_class=args.samples_per_class,
            seed=args.seed + i_repeat,
        )
        pool_indices = make_pool_indices(len(x_train), initial_indices)

        committee_members = [
            ActiveLearner(
                estimator=build_model(args.seed + i_repeat * args.committee_size + member_idx, args),
                X_training=x_train[initial_indices],
                y_training=y_train[initial_indices],
            )
            for member_idx in range(args.committee_size)
        ]

        committee = Committee(
            learner_list=committee_members,
            query_strategy=vote_entropy_sampling,
        )

        max_queries = min(args.n_queries, len(pool_indices))
        for i_query in tqdm(
            range(1, max_queries + 1),
            desc=f"QBC CNN repeat {i_repeat + 1}/{args.n_repeats}",
            leave=False,
        ):
            query_idx, _ = committee.query(x_train[pool_indices])
            rel_idx = int(np.asarray(query_idx).reshape(-1)[0])
            selected_idx = int(pool_indices[rel_idx])

            committee.teach(
                X=x_train[selected_idx : selected_idx + 1],
                y=y_train[selected_idx : selected_idx + 1],
            )

            if i_repeat == 0:
                first_run_queried_labels.append(int(y_train[selected_idx]))

            pool_indices = np.delete(pool_indices, rel_idx)
            score = committee.score(x_val, y_val)
            committee_results.append(ResultsRecord(f"committee_cnn_{args.committee_size}", i_query, score))

        final_test_scores.append(float(committee.score(x_test, y_test)))

    return committee_results, first_run_queried_labels, final_test_scores


def plot_learning_curves(
    random_summary: dict[str, list[float]],
    qbc_summary: dict[str, list[float]],
    output_path: Path,
    title: str,
) -> None:
    rounds = np.asarray(random_summary["query_ids"])

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, random_summary["mean_accuracy"], label="Random sampling + CNN", color="tab:orange")
    plt.fill_between(
        rounds,
        np.asarray(random_summary["mean_accuracy"]) - np.asarray(random_summary["std_accuracy"]),
        np.asarray(random_summary["mean_accuracy"]) + np.asarray(random_summary["std_accuracy"]),
        color="tab:orange",
        alpha=0.2,
    )
    plt.plot(rounds, qbc_summary["mean_accuracy"], label="QBC + CNN", color="tab:blue")
    plt.fill_between(
        rounds,
        np.asarray(qbc_summary["mean_accuracy"]) - np.asarray(qbc_summary["std_accuracy"]),
        np.asarray(qbc_summary["mean_accuracy"]) + np.asarray(qbc_summary["std_accuracy"]),
        color="tab:blue",
        alpha=0.2,
    )
    plt.xlabel("Query round")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QBC with a CNN committee on the Greek dataset")
    parser.add_argument("--committee-size", type=int, default=COMMITTEE_SIZE)
    parser.add_argument("--n-queries", type=int, default=N_QUERIES)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    x, y = load_greek_dataset(GREEK_DIR, img_size=args.img_size)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=args.seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VAL_SIZE,
        stratify=y_train_val,
        random_state=args.seed,
    )

    random_results, random_test_scores = run_random_baseline(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        args=args,
    )
    committee_results, queried_labels, qbc_test_scores = run_qbc_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        args=args,
    )

    random_summary = aggregate_records(random_results)
    qbc_summary = aggregate_records(committee_results)

    final_random = random_summary["mean_accuracy"][-1]
    final_qbc = qbc_summary["mean_accuracy"][-1]
    final_random_test = float(np.mean(random_test_scores))
    final_qbc_test = float(np.mean(qbc_test_scores))
    best_gap = max(
        qbc - baseline
        for qbc, baseline in zip(qbc_summary["mean_accuracy"], random_summary["mean_accuracy"])
    )

    plot_path = RESULTS_DIR / f"qbc_cnn_k{args.committee_size}.png"
    json_path = RESULTS_DIR / f"qbc_cnn_k{args.committee_size}.json"
    title = f"QBC + CNN vs Random + CNN (committee={args.committee_size})"

    plot_learning_curves(random_summary, qbc_summary, plot_path, title)

    summary = {
        "config": {
            "seed": args.seed,
            "img_size": args.img_size,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "n_queries": args.n_queries,
            "n_repeats": args.n_repeats,
            "committee_size": args.committee_size,
            "epochs_per_fit": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "samples_per_class": args.samples_per_class,
            "base_model": "CNN",
            "query_strategy": "vote_entropy_sampling",
        },
        "random": random_summary,
        "qbc": qbc_summary,
        "final_test_accuracy": {
            "random_mean": final_random_test,
            "qbc_mean": final_qbc_test,
        },
        "report_notes": {
            "final_random_validation_accuracy": final_random,
            "final_qbc_validation_accuracy": final_qbc,
            "final_random_test_accuracy": final_random_test,
            "final_qbc_test_accuracy": final_qbc_test,
            "best_qbc_minus_random_gap_on_validation": best_gap,
            "qbc_wins_final_round_on_validation": final_qbc > final_random,
        },
        "queried_labels_first_repeat": queried_labels[:15],
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Summary")
    print(f"  Final random val accuracy: {final_random:.4f}")
    print(f"  Final QBC val accuracy:    {final_qbc:.4f}")
    print(f"  Final random test accuracy:{final_random_test:.4f}")
    print(f"  Final QBC test accuracy:   {final_qbc_test:.4f}")
    print(f"  Best QBC val gap:          {best_gap:.4f}")
    print(f"  Plot saved to:             {plot_path}")
    print(f"  JSON saved to:             {json_path}")


if __name__ == "__main__":
    main()
