import json
import numpy as np
import matplotlib.pyplot as plt

with open('bo_results.json') as f:
    results = json.load(f)

max_acc_per_iter = results['max_acc_per_iter']
bo_best_per_iter = np.maximum.accumulate(-np.array(results['bo_func_vals'])).ravel()
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