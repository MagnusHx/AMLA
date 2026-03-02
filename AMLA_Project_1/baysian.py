import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import ParameterSampler
import skopt
from skopt import gp_minimize

from model import Model
from data import get_data_loaders
from train import train_model

# ----- Seed -----
random.seed(2118)
np.random.seed(2118)
torch.manual_seed(2118)

# ── Config ────────────────────────────────────────────────────────────────────
GREEK_DIR   = Path(__file__).parent / 'Greek'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS  = 20
NUM_CLASSES = 24
TRAIN_SPLIT = 0.8
IMG_SIZE    = 105

# ── Random search domain ──────────────────────────────────────────────────────
domain_random = {
    'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
    'dropout':       [0.05, 0.1, 0.2, 0.5, 0.8],
    'batch_size':    [32, 64, 128],
    'optimizer':     ['adam', 'sgd'],
    'criterion':     ['CrossEntropyLoss', 'MultiMarginLoss'],
}

# ── Helper: build criterion ───────────────────────────────────────────────────
def get_criterion(name):
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MultiMarginLoss':
        return nn.MultiMarginLoss()
    else:
        raise ValueError(f"Unknown criterion: {name}")

# ── Helper: build optimizer ───────────────────────────────────────────────────
def get_optimizer(name, model_params, lr):
    if name == 'adam':
        return torch.optim.Adam(model_params, lr=lr)
    elif name == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

# ── Core evaluation function ──────────────────────────────────────────────────
def evaluate_hyperparams(learning_rate, dropout, batch_size, optimizer_name, criterion_name):
    """Build model, optimizer and criterion, then delegate to train_model."""
    train_loader, val_loader = get_data_loaders(
        GREEK_DIR,
        batch_size=batch_size,
        train_split=TRAIN_SPLIT,
        img_size=IMG_SIZE
    )

    model     = Model(in_channels=1, num_classes=NUM_CLASSES, dropout=dropout)
    loss_fn   = get_criterion(criterion_name)
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)

    best_val_acc = train_model(
        model, train_loader, val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
    )

    return best_val_acc  # fraction 0-1

# ── Random Search ─────────────────────────────────────────────────────────────
print("=" * 60)
print("RANDOM SEARCH")
print("=" * 60)

param_list = list(ParameterSampler(domain_random, n_iter=20, random_state=32))

current_best     = 0.0
best_iter        = 0
max_acc_per_iter = []

for i, params in enumerate(param_list):
    print(f"\nIteration {i}: {params}")
    start = time.time()

    val_acc = evaluate_hyperparams(
        learning_rate  = params['learning_rate'],
        dropout        = params['dropout'],
        batch_size     = params['batch_size'],
        optimizer_name = params['optimizer'],
        criterion_name = params['criterion'],
    )

    print(f"Val accuracy: {val_acc:.4f} | Time: {time.time() - start:.1f}s")

    if val_acc > current_best:
        current_best = val_acc
        best_iter    = i

    max_acc_per_iter.append(current_best)

print(f"\nRandom search best: {current_best:.4f} (iteration {best_iter})")

# ── Bayesian Optimization ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BAYESIAN OPTIMIZATION")
print("=" * 60)

x0 = [
    param_list[0]['learning_rate'],
    param_list[0]['dropout'],
    param_list[0]['batch_size'],
    param_list[0]['optimizer'],
    param_list[0]['criterion'],
]
y0 = -max_acc_per_iter[0]

# Domain (plain tuples, same style as notebook)
learning_rate = (1e-4, 1e-1)
dropout       = (0.05, 0.8)
batch_size    = (32, 128)                            # cast to int inside objective
optimizer     = ('adam', 'sgd')
criterion     = ('CrossEntropyLoss', 'MultiMarginLoss')

global bo_iter
bo_iter = 1

def objective_function(x):
    global bo_iter
    lr        = x[0]
    drop      = x[1]
    bs        = int(round(x[2]))
    opt_name  = x[3]
    crit_name = x[4]

    print(f"\nBO iteration {bo_iter}: lr={lr:.5f}, dropout={drop:.3f}, "
          f"batch={bs}, opt={opt_name}, criterion={crit_name}")
    bo_iter += 1

    val_acc = evaluate_hyperparams(lr, drop, bs, opt_name, crit_name)
    print(f"Val accuracy: {val_acc:.4f}")
    return -val_acc

opt = gp_minimize(
    objective_function,
    [learning_rate, dropout, batch_size, optimizer, criterion],
    acq_func='EI',
    n_initial_points=0,
    n_calls=19,
    x0=[x0],
    y0=[y0],
    xi=0.1,
    noise=0.01**2,
    random_state=32
)

# ── Results ───────────────────────────────────────────────────────────────────
bo_best_per_iter = np.maximum.accumulate(-opt.func_vals).ravel()

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Random search best accuracy : {current_best:.4f}")
print(f"Bayesian optimization best  : {bo_best_per_iter[-1]:.4f}")
print(f"\nBO best hyperparameters:")
print(f"  learning_rate : {opt.x[0]:.6f}")
print(f"  dropout       : {opt.x[1]:.4f}")
print(f"  batch_size    : {int(round(opt.x[2]))}")
print(f"  optimizer     : {opt.x[3]}")
print(f"  criterion     : {opt.x[4]}")

# ── Save results ─────────────────────────────────────────────────────────────
import json
results = {
    'max_acc_per_iter': max_acc_per_iter,
    'bo_func_vals': opt.func_vals.tolist(),
    'bo_best_x': [x if isinstance(x, str) else int(x) if hasattr(x, '__index__') else float(x) for x in opt.x],
}
with open('bo_results.json', 'w') as f:
    json.dump(results, f)
print("Results saved to bo_results.json")

# ── Plot: Random Search vs Bayesian Optimization ──────────────────────────────
import matplotlib.pyplot as plt

iterations = range(1, len(max_acc_per_iter) + 1)

plt.figure()
plt.plot(iterations, max_acc_per_iter, 'r.-', label='Random Search')
plt.plot(iterations, bo_best_per_iter, 'b.-', label='Bayesian Optimization')
plt.xlabel('Iterations')
plt.ylabel('Best validation accuracy')
plt.title('Comparison between Random Search and Bayesian Optimization')
plt.legend()
plt.savefig('bo_vs_random_search.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to bo_vs_random_search.png")