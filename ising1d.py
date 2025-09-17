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


def run_ising1d(T:float, N:int=100, neq:int|None=None, k:float=1, epsilon:float=1, 
                navg:int=1, nsamples:int=1, evaluate_functions:list[Callable]=[]):
    # Give each dipole 1000 chances to flip by default
    # if not neq:
    #     neq = 1000 * N

    beta = 1 / (k*T)

    # tic = time.perf_counter()
    # Initialise state vector in random configuration of 1 and -1
    state = np.random.randint(2,size=(nsamples,N)) * 2 - 1

    # Initialise array of random dipole choices
    random_dipoles = np.random.randint(N, size=(nsamples,neq+navg))

    # Initialise random array for flip probabilities
    random_flip = np.random.rand(nsamples,neq+navg)
    # toc = time.perf_counter()
    # print(f"Time to initialise random samples: {toc - tic:.8f} s")

    # Iterate appropriate number of times to reach equilibrium
    for i in range(neq):
        for j in range(nsamples):
            run_iteration1d(state[j,:], random_dipoles[j,i], random_flip[j,i], N, beta, epsilon)
    
    func_results = [np.zeros((nsamples,navg)) for f in evaluate_functions]
    for i in range(navg):
        for j in range(nsamples):
            run_iteration1d(state[j,:], random_dipoles[j,i+neq], random_flip[j,i+neq], N, beta, epsilon)
            for (k,function) in enumerate(evaluate_functions):
                func_results[k][j,i] = function(state[j,:])
    
    func_results_averaged = [np.mean(fr,axis=1) for fr in func_results]
    return state, *func_results_averaged


def run_iteration1d(state, random_dipole, random_flip, N, beta, epsilon):
    # Compute energy difference of flipping chosen dipole
    # DeltaU = compute_energy1d(new_state, epsilon) - compute_energy1d(state, epsilon)
    DeltaU = 2*epsilon * state[random_dipole] * (state[(random_dipole + 1) % N] + state[(random_dipole - 1)%N])

    # Flip spin with probabiligy exp(-beta*DeltaU)
    if random_flip < np.exp(-beta * DeltaU):
        state[random_dipole] = -state[random_dipole]

# def run_iteration1d(state, random_dipole, random_flip, N, beta, epsilon):
#     # Compute energy difference of flipping chosen dipole
#     # DeltaU = compute_energy1d(new_state, epsilon) - compute_energy1d(state, epsilon)
#     DeltaU = 2*epsilon * state[random_dipole] * (state[(random_dipole + 1) % N] + state[(random_dipole - 1)%N])
#     # Flip spin of chosen dipole if energy difference is negative
#     state[random_dipole] = (~(random_flip < np.exp(-beta * DeltaU))*2-1) * state[random_dipole]
#     # if DeltaU < 0:
#     #     state[random_dipole] = -state[random_dipole]
    
#     # # Otherwise, flip spin with probabiligy exp(-beta*DeltaU)
#     # elif random_flip < np.exp(-beta * DeltaU):
#     #     state[random_dipole] = -state[random_dipole]
    

# def run_ising1d(T:float, N:int=100, neq:int|None=None, k:float=1, epsilon:float=1, 
#                 navg:int = 10000, nsamples:int=1, evaluate_functions:list[Callable]=[]):
#     # Give each dipole 1000 chances to flip by default
#     if not neq:
#         neq = 1000 * N

#     beta = 1 / (k*T)

#     # Initialise state vector in random configuration of 1 and -1
#     state = np.random.randint(2,size=(nsamples, N)) * 2 - 1
#     # Initialise array of random dipole choices
#     random_dipoles = np.random.randint(N, size=(nsamples, neq+navg))

#     # Initialise random array for flip probabilities
#     random_flip = np.random.rand(nsamples, neq+navg)

#     # Iterate appropriate number of times to reach equilibrium
#     for i in range(neq):
#         run_iteration1d(state, random_dipoles[:,i], random_flip[:,i], N, nsamples, beta, epsilon)
    
#     # Compute time averaged values of functions
#     func_results = [np.zeros((nsamples, navg)) for f in evaluate_functions]
#     for i in range(navg):
#         run_iteration1d(state, random_dipoles[:,i+neq], random_flip[:,i+neq], N, nsamples, beta, epsilon)
#         for (j,function) in enumerate(evaluate_functions):
#             func_results[j][:,i] = function(state)
        
#     return state, *func_results


# def run_iteration1d(state, chosen_dipole, random_flip, N, nsamples, beta, epsilon):
#     # Compute energy difference of flipping chosen dipole
#     # DeltaU = compute_energy1d(new_state, epsilon) - compute_energy1d(state, epsilon)
#     # rows = np.arange(nsamples)
#     flat_idx = np.arange(nsamples) * N + chosen_dipole
#     DeltaU = 2*epsilon * state.flat[flat_idx] * (state.flat[flat_idx + 1 - N * ((chosen_dipole + 1))// N] + state.flat[flat_idx - 1 - N *((chosen_dipole - 1)//N)])
#     # Flip spin of chosen dipole if energy difference is negative or randomly flip according to boltzmann factor
#     flip = (random_flip < np.exp(-beta * DeltaU)) * -2 + 1

#     state.flat[flat_idx] = flip * state.flat[flat_idx]

