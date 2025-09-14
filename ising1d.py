import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import time

def compute_energy1d(state, epsilon=1):
    return - epsilon * np.sum(state * np.roll(state, 1))

def compute_magnetisation1d(state, mu_B=1):
    return mu_B * np.sum(state)

# def run_ising1d(T:float, N:int=100, niters:int|None=None, k:float=1, epsilon:float=1, 
#                 nsamples:int=1, evaluate_functions:list[Callable]=[]):
#     # Give each dipole 1000 chances to flip by default
#     if not niters:
#         niters = 1000 * N

#     beta = 1 / (k*T)

#     # Initialise state vector in random configuration of 1 and -1
#     state = np.random.randint(2,size=(N)) * 2 - 1

#     # Iterate appropriate number of times to reach equilibrium
#     for i in range(niters):
#         state = run_iteration1d(state, beta, epsilon)

    
#     states = np.zeros((nsamples,N))
#     func_results = [np.zeros(nsamples) for f in evaluate_functions]
    
#     states[0,:] = state
#     for (j,function) in enumerate(evaluate_functions):
#         func_results[j][0] = function(states[0,:])

#     for i in range(nsamples-1):
#         states[i+1,:] = run_iteration1d(states[i,:], beta, epsilon)
#         for (j,function) in enumerate(evaluate_functions):
#             func_results[j][i+1] = function(states[i+1,:])
        
#     return states, *func_results


# def run_iteration1d(state, beta, epsilon):
#     # Choose a dipole at random
#     chosen_dipole = np.random.randint(len(state))

#     # State with chosen dipole flipped
#     new_state = np.array(state)
#     new_state[chosen_dipole] = -state[chosen_dipole]

#     # Compute energy difference of flipping chosen dipole
#     DeltaU = compute_energy1d(new_state, epsilon) - compute_energy1d(state, epsilon)

#     # Flip spin of chosen dipole if energy difference is negative
#     if DeltaU < 0:
#         return new_state
    
#     # Otherwise, flip spin with probabiligy exp(-beta*DeltaU)
#     else:
#         pflip = np.exp(-beta * DeltaU)
#         if np.random.rand() < pflip:
#             return new_state
    
#     # Return the initial state if not flipped
#     return state


def run_ising1d(T:float, N:int=100, niters:int|None=None, k:float=1, epsilon:float=1, 
                nsamples:int=1, evaluate_functions:list[Callable]=[]):
    # Give each dipole 1000 chances to flip by default
    if not niters:
        niters = 1000 * N

    beta = 1 / (k*T)

    # tic = time.perf_counter()
    # Initialise state vector in random configuration of 1 and -1
    state = np.random.randint(2,size=(N)) * 2 - 1

    # Initialise array of random dipole choices
    random_dipoles = np.random.randint(N, size=(niters+nsamples))

    # Initialise random array for flip probabilities
    random_flip = np.random.rand(niters+nsamples)
    # toc = time.perf_counter()
    # print(f"Time to initialise random samples: {toc - tic:.8f} s")

    # Iterate appropriate number of times to reach equilibrium
    for i in range(niters):
        run_iteration1d(state, random_dipoles[i], random_flip[i], N, beta, epsilon)
    
    func_results = [np.zeros(nsamples) for f in evaluate_functions]
    for i in range(nsamples):
        run_iteration1d(state, random_dipoles[i+niters], random_flip[i+niters], N, beta, epsilon)
        for (j,function) in enumerate(evaluate_functions):
            func_results[j][i] = function(state)
        
    return state, *func_results


def run_iteration1d(state, random_dipole, random_flip, N, beta, epsilon):
    # Compute energy difference of flipping chosen dipole
    # DeltaU = compute_energy1d(new_state, epsilon) - compute_energy1d(state, epsilon)
    DeltaU = 2*epsilon * state[random_dipole] * (state[(random_dipole + 1) % N] + state[(random_dipole - 1)%N])
    # Flip spin of chosen dipole if energy difference is negative
    if DeltaU < 0:
        state[random_dipole] = -state[random_dipole]
    
    # Otherwise, flip spin with probabiligy exp(-beta*DeltaU)
    elif random_flip < np.exp(-beta * DeltaU):
        state[random_dipole] = -state[random_dipole]
    