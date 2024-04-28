from utils.rlc_pinn import apply_RLC_PINN
from itertools import product
from tqdm import tqdm

# TODO: implemente analytical solution as python function and add noise to observed data
# TODO: add boundary loss condition and respective changes on graphs in PINN class and utils graphs
# TODO: Maybe integrate Matlab too ? 

# For more information check ==> https://github.com/jmorrow1000/PINN-iPINN/tree/main
if __name__ == "__main__": 


    R = 1.2
    L = 1.5
    C = 0.3
    RL_gt = R/L # R/L = 0,8
    LC_gt = L*C # 1/LC = 2,222

    points = [50]
    guesses = [[0.0,0.0]]
    lambdas = [10**4]
    adam_iter = [3*10**3]
    LBFGSs = [True]
    Layers = [3]
    Neurons = [32]
    ALL = list(product(points, guesses, lambdas, adam_iter, LBFGSs, Layers, Neurons))

    for (point, gues, lambd, adam_ite, LBFGS, Layer, Neuron) in tqdm(ALL):
        apply_RLC_PINN( R, L, C, training_points = point,
                        guess = gues, lbd = lambd, iterations = adam_ite, LBFGS = LBFGS,
                        layers = Layer, neurons = Neuron,
                        DATA_PATH = r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-RLC", 
                        GIF_FIGS = 99,
                        SHOW_ITER = 50,
                        SHOW_PRINTS = True,
                        SHOW_MODEL = False,
                        SAVE_RESULTS = True)









    # R = 1.2
    # L = 1.5
    # C = 0.3
    # RL_gt = R/L # R/L = 0,8
    # LC_gt = L*C # 1/LC = 2,222

    # points = [10,50]
    # guesses = [[-20.0,-10.0], [0.0,0.0], [10.0,10.0], [-10.0,20.0], [10.0,100.0],]
    # lambdas = [10**2, 10**4]
    # adam_iter = [6*10**2, 3*10**3]
    # LBFGSs = [True, False]
    # # Layers = [3, 6, 9]
    # # Neurons = [8, 16, 32, 64]
    # Layers = [3, 9]
    # Neurons = [8, 32]
    # ALL = list(product(points, guesses, lambdas, adam_iter, LBFGSs, Layers, Neurons))

    # print(f"\n\n\t\t\t Starting Monte Carlo Grid Simulation of {len(ALL)} scenarios, \n\t\t\t\t\tGRAB SOME COFFEE ! \n\n")
    # for (point, gues, lambd, adam_ite, LBFGS, Layer, Neuron) in tqdm(ALL):
    #     apply_RLC_PINN( R, L, C, training_points = point,
    #                     guess = gues, lbd = lambd, iterations = adam_ite, LBFGS = LBFGS,
    #                     layers = Layer, neurons = Neuron,
    #                     DATA_PATH = r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-RLC", 
    #                     GIF_FIGS = 99,
    #                     SHOW_ITER = 50,
    #                     SHOW_PRINTS = False,
    #                     SHOW_MODEL = False,
    #                     SAVE_RESULTS = True)