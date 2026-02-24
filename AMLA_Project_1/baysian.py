import skopt
from skopt import gp_minimize
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
import time

# hyperparams dictionary 
domain = 

# create the ParameterSampler
param_list = list(ParameterSampler(domain, n_iter=20, random_state=32))
print('Param list')
print(param_list)

## now we can train the random forest using these parameters tuple, and for
## each iteration we store the best value of the oob

current_best_oob = 0
iteration_best_oob = 0 
max_oob_per_iteration = []
i = 0
for params in param_list:
    print(i)
    print(params)
    
    #define model here
    
    start = time.time()
    #train the model

    end = time.time()
    # extract oob_score and update current_best_oob if better that the current best
    
    max_oob_per_iteration.append(current_best_oob)
    i += 1
    print(f'It took {end - start} seconds')
    

#start at same initial point
x0=[param_list[0]['n_estimators'],param_list[0]['max_depth'],param_list[0]['max_features'],param_list[0]['criterion']]
if x0[2] is None:
    x0[2] = 'None'


y0 = -max_oob_per_iteration[0]


## define the domain of the considered parameters
n_estimators = (1,150)
max_depth=(1,100)
max_features = ('log2', 'sqrt', 'None')
criterion = ('gini', 'entropy')


## we have to define the function we want to maximize --> validation accuracy, 
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
global i
i = 1
def objective_function(x): 
    if x[2]=='None':
        maxf = None
    else:
        maxf = x[2]
    
    #create the model
    
    # fit the model 
    
    global i
    i += 1
    print(i)
    print(x)
    print(model.oob_score_)
    
    return - model.oob_score_

np.int = int #numpy np.int deprecation workaround
opt = gp_minimize()