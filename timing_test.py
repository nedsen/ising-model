import numpy as np
import matplotlib.pyplot as plt
from ising1d import compute_energy1d, run_ising1d_faster, compute_magnetisation1d, run_iteration1d_faster


import time

n = 10

tic = time.time()

for i in range(n):
    state = run_ising1d_faster(N=100, T=0.5, niters=200000)


    # for i in range(100000):
    #     r = np.random.rand()

    # np.random.rand(100000)

    # state = run_iteration1d_faster

toc = time.time()

print(f"Elapsed time: {toc - tic:.4f} s")
print(f"Average time: {(toc - tic)/n:.4f} s")
