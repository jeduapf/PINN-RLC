from utils import *
import sys

def apply_RLC_PINN( R = 1.2, L = 1.5, C = 0.3, training_points = 150,
                    guess = [10.0, -8.0], lbd = 10**3, iterations = 2000, LBFGS = True,
                    layers = 8, neurons = 20,
                    DATA_PATH = r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-RLC", 
                    GIF_FIGS = 99,
                    SHOW_ITER = 50,
                    SHOW_PRINTS = False,
                    SHOW_MODEL = False,
                    SAVE_RESULTS = True):  
    
    SAVE_DIR, SAVE_DIR_GIF = save_paths(f"layers_{layers}__neurons_{neurons}__RLC__{R:.2f}_{L:.2f}_{C:.2f}_Iter_{iterations}__Points_{training_points}__Lambda_{lbd}__RL_{guess[0]:.2f}__LC_{guess[1]:.2f}")
    
    # ****************************** Initializating the model ******************************

    X_star, u_star = import_matlab_data(out = "u.mat", inp = "sin.mat")

    N_u = training_points   # Number of randomly observed points
    iterations = iterations # Adam descending steps
    lambd = lbd             # Weight over data loss compared to physics loss
    chute = guess           # R/L  = 0.8, 1/L*C = 2.22 

    X_train, u_train, idx = choose_points(N_u, X_star, u_star)

    # ****************************** Training the model ******************************

    model = PhysicsInformedNN(  X_star, u_star, 
                                X_train, u_train,  
                                R,
                                L,
                                C,
                                inputs = 1, outputs = 1, 
                                guess = chute,
                                layer = layers, neuron = neurons,
                                SHOW_MODEL = SHOW_MODEL,
                                SHOW_PRINTS = SHOW_PRINTS, 
                                SHOW_ITER = SHOW_ITER,
                                GIF_FIGS = GIF_FIGS,
                                SAVE_DIR_GIF = SAVE_DIR_GIF)
    model.train(nIter = iterations, LBFGS = LBFGS, u_f = lambd)

    # ****************************** Evaluating the model ******************************
    
    # Predict over whole data
    string = model.evaluate()

    # ****************************** Saving results ******************************
    
    if SAVE_RESULTS:
        if SHOW_PRINTS:
            print(f'\n\n\t\tSaving Results\nIter: {iterations}\nPoints: {training_points}\nLambda: {lbd}\nR/L: {guess[0]}\nL*C: {guess[1]}\n\t\t\t.')
        save_txt(string, os.path.join(SAVE_DIR,"results.txt"))
        plot_ground_truth(X_star, u_star, SAVE_DIR)
        loss_plot(model, SAVE_DIR)
        plot_final_prediction(model, X_star, u_star, SAVE_DIR)
        if SHOW_PRINTS:
            print(f'\t\t\t.')
        save_gif_PIL(os.path.join(SAVE_DIR,"learning.gif"), model.files, fps=5, loop=0)
        if SHOW_PRINTS:
            print(f'\t\t\t.')