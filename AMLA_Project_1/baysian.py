import skopt
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from data import get_data_loaders
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
import time
import matplotlib.pyplot as plt
from model import Model

# device and data loaders (use same dataset as main.py)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
greek_dir = Path(__file__).resolve().parent / 'Greek'
# default batch_size here will be overridden in objective if needed
train_loader, val_loader = get_data_loaders(greek_dir, batch_size=32, train_split=0.8, img_size=105)

# hyperparams dictionary 
domain = {#'kernel size': [3, 5, 7],
          #'optimizer': ['adam', 'sgd'],
          'learning rate': [0.001, 0.01, 0.1],
          'batch size': [32, 64, 128],
          'dropout': [0.05, 0.1, 0.5, 0.8],
          'criterion': ['crossentropy', 'multimargin']}

# create the ParameterSampler
param_list = list(ParameterSampler(domain, n_iter=20, random_state=32))
print('Param list')
print(param_list)

## now we can train the random forest using these parameters tuple, and for
## each iteration we store the best value of the oob

current_best_oob = 0
iteration_best_oob = 0 
max_oob_per_iteration = []
loop_i = 0
for params in param_list:
    print(loop_i)
    print(params)
    
    #define model here
    model = Model()
    start = time.time()
    #train the model

    end = time.time()
    # extract oob_score and update current_best_oob if better that the current best
    
    max_oob_per_iteration.append(current_best_oob)
    loop_i += 1
    print(f'It took {end - start} seconds')
    

#start at same initial point
def _criterion_name(c):
    if c is None:
        return 'None'
    if isinstance(c, str):
        return c
    # match by class name
    cname = c.__class__.__name__.lower()
    if 'crossent' in cname:
        return 'crossentropy'
    if 'multimargin' in cname:
        return 'multimargin'
    return 'None'

x0 = [
    param_list[0]['learning rate'],
    param_list[0]['dropout'],
    _criterion_name(param_list[0].get('criterion')),
    param_list[0]['batch size'],
]


y0 = -max_oob_per_iteration[0]


## define the domain of the considered parameters
learning_rate = (0.001, 0.1)
dropout = (0.05, 0.8)
learning_rate = Real(1e-6, 1e-1, prior='log-uniform')
dropout = Real(0.05, 0.8)
# use simple string labels for skopt categorical dimension
criterion = Categorical(['crossentropy', 'multimargin'])
batch_size = Categorical([32, 64, 128])



## we have to define the function we want to maximize --> validation accuracy, 
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
# counter for objective invocations
i = 1
def objective_function(x): 
    # map selected criterion string to actual loss
    if x[2] == 'None':
        selected_criterion = None
    elif x[2] == 'crossentropy':
        selected_criterion = nn.CrossEntropyLoss()
    elif x[2] == 'multimargin':
        selected_criterion = nn.MultiMarginLoss()
    else:
        selected_criterion = None
    
    # create fresh data loaders if batch size changes
    bs = int(x[3])
    local_train_loader, local_val_loader = get_data_loaders(greek_dir, batch_size=bs, train_split=0.8, img_size=105)

    # create the model
    model = Model(learning_rate=x[0], dropout=float(x[1]), criterion=selected_criterion, batch_size=bs)
    # fit the model using the chosen learning rate
    model = model.fit(
            local_train_loader,
            val_loader=local_val_loader,
            num_epochs=10,
            learning_rate=float(x[0]),
            device=device
        )

    global i
    i += 1
    val_loss, val_acc = model.evaluate(local_val_loader, device=device)
    print(f'iter={i} x={x} -> val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

    # we want to maximize val_acc, so minimize -val_acc
    return -float(val_acc)

np.int = int #numpy np.int deprecation workaround
opt = gp_minimize(
    objective_function,  # the function to minimize
    [learning_rate, dropout, criterion, batch_size],
    # the bounds on each dimension of x
    acq_func="EI",  # the acquisition function
    n_initial_points=5,  # add a few random initial points to explore
    n_calls=19,  # total number of evaluations (kept modest to limit runtime)
    x0=[x0],  # initial point
    y0=[y0],  # initial objective function value
    xi=0.1,  # exploration parameter
    noise=0.01**2,  # the noise level (optional)
    random_state=32,
)

# print results
print('\nOptimization finished')
print('best encoded x:', opt.x)
print('best objective (neg val acc):', opt.fun)
# if criterion is categorical string it's already readable
best = {
    'learning_rate': opt.x[0],
    'dropout': opt.x[1],
    'criterion': opt.x[2],
    'batch_size': int(opt.x[3])
}
print('best hyperparameters:', best)

# collect the maximum each iteration of BO
y_bo = np.maximum.accumulate(-opt.func_vals).ravel()

print(y_bo)
# define iteration number
xs = np.arange(1,21,1)

plt.plot(xs, max_oob_per_iteration, 'o-', color = 'red', label='Random Search')
plt.plot(xs, y_bo, 'o-', color = 'blue', label='Bayesian Optimization')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Out of bag error')
plt.title('Comparison between Random Search and Bayesian Optimization')
plt.show()