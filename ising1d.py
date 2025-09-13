import numpy as np
import matplotlib.pyplot as plt


def compute_energy1d(state, epsilon=1):
    return - epsilon * np.sum(state * np.roll(state, 1))

def run_ising1d(T:float, N:int=100, niters:int|None=None, k:float=1, epsilon:float=1):
    # Give each dipole 1000 chances to flip by default
    if not niters:
        niters = 1000 * N

    beta = 1 / (k*T)

    # Initialise state vector in random configuration of 1 and -1
    state = np.random.randint(2,size=(N)) * 2 - 1

    for i in range(niters):
        # Choose a dipole at random
        chosen_dipole = np.random.randint(N)

        # State with chosen dipole flipped
        new_state = np.array(state)
        new_state[chosen_dipole] = -state[chosen_dipole]

        # Compute energy difference of flipping chosen dipole
        DeltaU = compute_energy1d(new_state, epsilon) - compute_energy1d(state, epsilon)

        # Flip spin of chosen dipole if energy difference is negative
        if DeltaU < 0:
            state = new_state
        
        # Otherwise, flip spin with probabiligy exp(-beta*DeltaU)
        else:
            pflip = np.exp(-beta * DeltaU)
            if np.random.rand() < pflip:
                state = new_state

    return state

def run_iteration1d(state, beta, epsilon):
    # Choose a dipole at random
    chosen_dipole = np.random.randint(len(state))

    # State with chosen dipole flipped
    new_state = np.array(state)
    new_state[chosen_dipole] = -state[chosen_dipole]

    # Compute energy difference of flipping chosen dipole
    DeltaU = compute_energy1d(new_state, epsilon) - compute_energy1d(state, epsilon)

    # Flip spin of chosen dipole if energy difference is negative
    if DeltaU < 0:
        return new_state
    
    # Otherwise, flip spin with probabiligy exp(-beta*DeltaU)
    else:
        pflip = np.exp(-beta * DeltaU)
        if np.random.rand() < pflip:
            return new_state
    
    # Return the initial state if not flipped
    return state
