from utils.rlc_pinn import apply_RLC_PINN
from utils.utils import exact_solution
import numpy as np

# TODO: add boundary loss condition and respective changes on graphs in PINN class and utils graphs

# For more information check ==> https://github.com/jmorrow1000/PINN-iPINN/tree/main
if __name__ == "__main__": 

    # Real System Definitions
    L = 1.5         # Henries
    C = 0.3         # Faradays
    R = 1.2         # Ohms
    Vin = 1.0     # V
    t_init = 0      # s
    t_final = 5    # s
    points = 10**4  # -

    noise = 40              # db
    mu = 0                  # noise center
    sigma = np.sqrt(np.sqrt(Vin))    # noise standard deviation

    # Neural Network Hyperparameters
    training_points = 600 
    lbd = 10**3 
    iterations = 1000
    guess = [2.0, 2.0]

    # Neural Network Structure
    LBFGS = True
    layers = 3
    neurons = 64

    print(f"\n\n\t\t\tRessonant frequency of RLC at {np.sqrt(1/L/C)}\n\n")
    apply_RLC_PINN( t_init, t_final, points, training_points,
                    R , L, C , Vin , noise, mu, sigma,
                    guess, lbd, iterations, LBFGS,
                    layers, neurons,
                    GIF_FIGS = 99,
                    SHOW_ITER = 50,
                    SHOW_PRINTS = True,
                    SHOW_MODEL = False,
                    SAVE_RESULTS = True)
    
