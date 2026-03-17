## AMLA Project 1

This project now has three main workflows:

1. Standard supervised CNN training on the Greek character dataset.
2. Hyperparameter search for the CNN with random search vs Bayesian optimization.
3. Active learning experiments:
   `qbc_active_learning.py` uses a Random Forest committee.
   `qbc_cnn_active_learning.py` uses a CNN committee.

## How The Pipeline Works

### Dataset

The dataset lives in `Greek/`, with one folder per class.

- [`data.py`](/home/linux/repositories/AMLA/AMLA_Project_1/data.py) loads grayscale images, resizes them, and creates reproducible stratified train/validation splits for CNN training.
- The active-learning scripts load the same dataset directly and create:
  - a training pool
  - a validation set used during querying
  - a held-out test set used only once at the end

### CNN Training

- [`model.py`](/home/linux/repositories/AMLA/AMLA_Project_1/model.py) defines the CNN.
- [`train.py`](/home/linux/repositories/AMLA/AMLA_Project_1/train.py) contains the shared training loop, validation loop, optimizer builder, and loss-function builder.
- [`main.py`](/home/linux/repositories/AMLA/AMLA_Project_1/main.py) is the main supervised training entry point.

During training:

1. Images are loaded and resized.
2. The CNN is trained for a number of epochs.
3. Validation accuracy is tracked after each epoch.
4. The best checkpoint is saved to `best_model.pth`.

### Bayesian Optimization

- [`baysian.py`](/home/linux/repositories/AMLA/AMLA_Project_1/baysian.py) compares random search against Bayesian optimization for CNN hyperparameters.
- It tunes:
  - learning rate
  - dropout
  - batch size
  - optimizer
  - loss function

It stores outputs in `results/bo_results.json` and `results/bo_vs_random_search.png`.

### Active Learning

#### Random Forest QBC

- [`qbc_active_learning.py`](/home/linux/repositories/AMLA/AMLA_Project_1/qbc_active_learning.py)

This script:

1. Builds a train/validation/test split.
2. Starts with a tiny labeled set.
3. Compares:
   - random sampling
   - query-by-committee with vote entropy
4. Uses validation accuracy for the learning curve.
5. Uses the test set only for the final summary.

#### CNN QBC

- [`qbc_cnn_active_learning.py`](/home/linux/repositories/AMLA/AMLA_Project_1/qbc_cnn_active_learning.py)

This follows the same logic, but each committee member is a CNN instead of a Random Forest.

## What To Run

### 1. Install dependencies

```bash
uv sync
```

### 2. Train the CNN

```bash
.venv/bin/python main.py
```

Useful options:

```bash
.venv/bin/python main.py --epochs 15 --batch-size 32 --learning-rate 0.001 --dropout 0.2
```

### 3. Run Bayesian optimization

Full run:

```bash
.venv/bin/python baysian.py
```

Quick smoke test:

```bash
.venv/bin/python baysian.py --random-iters 2 --bo-iters 2 --epochs 1
```

### 4. Run Random Forest active learning

```bash
.venv/bin/python qbc_active_learning.py
```

Quick version:

```bash
.venv/bin/python qbc_active_learning.py --n-queries 10 --n-repeats 1
```

### 5. Run CNN active learning

```bash
.venv/bin/python qbc_cnn_active_learning.py
```

Quick version:

```bash
.venv/bin/python qbc_cnn_active_learning.py --n-queries 10 --n-repeats 1 --epochs 2 --committee-size 3
```

## Output Files

Most experiment outputs are written to `results/`.

- `qbc_modal_k*.png/json`: Random Forest active learning
- `qbc_cnn_k*.png/json`: CNN active learning
- `bo_results.json`: hyperparameter-search results
- `bo_vs_random_search.png`: hyperparameter-search plot

## Current Status

The main training script, Bayesian optimization script, and both QBC scripts are now aligned around a consistent pipeline.

- `main.py` trains the CNN correctly.
- `baysian.py` uses the real CNN training API instead of calling non-existent `.fit()` or `.evaluate()` methods on the model.
- The QBC scripts now evaluate on validation during querying and reserve the test set for final reporting.
