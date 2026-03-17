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
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"The number of unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
)

SEED = 1422
IMG_SIZE = 28
TEST_SIZE = 0.2
VAL_SIZE = 0.25
N_QUERIES = 75
N_REPEATS = 3
COMMITTEE_SIZE = 5
N_TREES = 20
SAMPLES_PER_CLASS = 1
RESULTS_DIR = Path(__file__).parent / "results"
GREEK_DIR = Path(__file__).parent / "Greek"

ModelClass = RandomForestClassifier
ResultsRecord = namedtuple("ResultsRecord", ["estimator", "query_id", "score"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_greek_dataset(root_dir: Path, img_size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    for class_idx, class_dir in enumerate(sorted(root_dir.glob("character*"))):
        for image_path in sorted(class_dir.glob("*.png")):
            image = Image.open(image_path).convert("L").resize((img_size, img_size))
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(image_array.reshape(-1))
            labels.append(class_idx)

    return np.asarray(images), np.asarray(labels)


def build_model(seed: int) -> RandomForestClassifier:
    return ModelClass(n_estimators=N_TREES, random_state=seed)


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
    n_queries: int,
    n_repeats: int,
    samples_per_class: int,
    seed: int,
) -> tuple[list[ResultsRecord], list[float]]:
    random_results = []
    final_test_scores = []

    for i_repeat in range(n_repeats):
        initial_indices = choose_initial_indices(y_train, samples_per_class=samples_per_class, seed=seed + i_repeat)
        pool_indices = make_pool_indices(len(x_train), initial_indices)
        rng = np.random.default_rng(seed + i_repeat)
        rng.shuffle(pool_indices)
        learner = build_model(seed + i_repeat)
        labeled_indices = initial_indices.copy()
        learner.fit(X=x_train[labeled_indices, :], y=y_train[labeled_indices])
        max_queries = min(n_queries, len(pool_indices))

        for i_query in tqdm(
            range(1, max_queries + 1),
            desc=f"Random baseline repeat {i_repeat + 1}/{n_repeats}",
            leave=False,
        ):
            new_index = pool_indices[i_query - 1]
            labeled_indices = np.append(labeled_indices, new_index)
            learner.fit(X=x_train[labeled_indices, :], y=y_train[labeled_indices])
            score = learner.score(x_val, y_val)
            random_results.append(ResultsRecord("random", i_query, score))

        final_test_scores.append(float(learner.score(x_test, y_test)))

    return random_results, final_test_scores


def run_qbc_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_queries: int,
    n_repeats: int,
    committee_size: int,
    samples_per_class: int,
    seed: int,
) -> tuple[list[ResultsRecord], list[int], list[float]]:
    try:
        from modAL.disagreement import vote_entropy_sampling
        from modAL.models import ActiveLearner, Committee
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "modAL is required for this script because the exercise uses "
            "ActiveLearner, Committee, and vote_entropy_sampling. "
            "Install it in the project environment before running this file."
        ) from exc

    committee_results = []
    first_run_queried_labels = []
    final_test_scores = []

    for i_repeat in range(n_repeats):
        initial_indices = choose_initial_indices(y_train, samples_per_class=samples_per_class, seed=seed + i_repeat)
        pool_indices = make_pool_indices(len(x_train), initial_indices)
        committee_members = [
            ActiveLearner(
                estimator=build_model(seed + i_repeat * committee_size + member_idx),
                X_training=x_train[initial_indices, :],
                y_training=y_train[initial_indices],
            )
            for member_idx in range(committee_size)
        ]

        committee = Committee(
            learner_list=committee_members,
            query_strategy=vote_entropy_sampling,
        )

        max_queries = min(n_queries, len(pool_indices))
        for i_query in tqdm(
            range(1, max_queries + 1),
            desc=f"QBC repeat {i_repeat + 1}/{n_repeats}",
            leave=False,
        ):
            query_idx, _ = committee.query(x_train[pool_indices])
            rel_idx = int(np.asarray(query_idx).reshape(-1)[0])
            selected_idx = int(pool_indices[rel_idx])

            committee.teach(
                X=x_train[selected_idx : selected_idx + 1],
                y=y_train[selected_idx : selected_idx + 1],
            )
            committee._set_classes()

            if i_repeat == 0:
                first_run_queried_labels.append(int(y_train[selected_idx]))

            pool_indices = np.delete(pool_indices, rel_idx)

            score = committee.score(x_val, y_val)
            committee_results.append(ResultsRecord(f"committee_{committee_size}", i_query, score))

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
    plt.plot(rounds, random_summary["mean_accuracy"], label="Random sampling", color="tab:orange")
    plt.fill_between(
        rounds,
        np.asarray(random_summary["mean_accuracy"]) - np.asarray(random_summary["std_accuracy"]),
        np.asarray(random_summary["mean_accuracy"]) + np.asarray(random_summary["std_accuracy"]),
        color="tab:orange",
        alpha=0.2,
    )
    plt.plot(rounds, qbc_summary["mean_accuracy"], label="QBC (modAL)", color="tab:blue")
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
    parser = argparse.ArgumentParser(
        description="QBC on the Greek dataset using the same modAL methods as the exercise notebook"
    )
    parser.add_argument("--committee-size", type=int, default=COMMITTEE_SIZE)
    parser.add_argument("--n-queries", type=int, default=N_QUERIES)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
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
        n_queries=args.n_queries,
        n_repeats=args.n_repeats,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    committee_results, queried_labels, qbc_test_scores = run_qbc_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        n_queries=args.n_queries,
        n_repeats=args.n_repeats,
        committee_size=args.committee_size,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
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

    plot_path = RESULTS_DIR / f"qbc_modal_k{args.committee_size}.png"
    json_path = RESULTS_DIR / f"qbc_modal_k{args.committee_size}.json"
    title = f"QBC vs Random on Greek Characters (modAL, committee={args.committee_size})"

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
            "samples_per_class": args.samples_per_class,
            "base_model": "RandomForestClassifier",
            "query_strategy": "vote_entropy_sampling",
            "implementation_source": "ActiveLearning_exercise8_modAL_solution.ipynb",
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
