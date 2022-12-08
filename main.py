import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt
from simulated_annealing import SimulatedAnnealing


"""
No need to change this cell.
"""

def read_inputs(path='inputs.pkl'):
    inputs = None
    with open(path, 'rb') as fp:
        inputs = pkl.load(fp)
    return inputs

inputs = read_inputs()

"""
No need to change this cell.
"""

def plot(records):
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Vs. Iterations")
    plt.plot(records['iteration'], records['best_cost'])
    plt.show()

#        for i in range(len(S)):
#            print(i)
#            print("###")
#            print(len(S))
#            if state[i] == "1":
#                sum += S[i]
#        return abs(sum - T)


simulated_annealing = SimulatedAnnealing()
for test in inputs:
    best_cost, best_solution, records = simulated_annealing.run_algorithm(test['S'], test['T'], stopping_iter=2*len(test['S']))
    print(f"Best Value Found: {np.dot(best_solution, test['S'])} - Target Value: {test['T']}")
    plot(records)