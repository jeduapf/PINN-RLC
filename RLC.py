from utils import *

if __name__ == "__main__": 
    DATA_PATH = r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-RLC"    
    SAVE_DIR, SAVE_DIR_GIF = save_paths()
    
    # ****************************** Initializating the model ******************************

    #          X_star       u_star
    #        [ t , v(t)]   [ i(t) ]
    #          t1 , v1        i1
    #          t2 , v2        i2
    # points     ...          ...
    #          tN , vN        iN
    X_star, u_star = import_matlab_data(out = "u.mat", inp = "sin.mat")

    # Number of randomly observed points
    N_u = 150
    # Adam descending steps
    iterations = 2000
    # Xeight over data loss compared to physics loss
    lambd = 10**3
    # R/L  = 0.8, 1/L*C = 2.22 
    chute = [10.0, -8.0]

    X_train, u_train, idx = choose_points(N_u, X_star, u_star)

    # ****************************** Training the model ******************************

    model = PhysicsInformedNN(  X_star, u_star, 
                                X_train, u_train,  
                                R = 1.2,
                                L = 1.5,
                                C = 0.3,
                                inputs = 1, outputs = 1, 
                                guess = chute,
                                SHOW_MODEL = False,
                                SHOW_PRINTS = True, 
                                SHOW_ITER = 50,
                                GIF_FIGS = 99,
                                SAVE_DIR_GIF = SAVE_DIR_GIF)
    model.train(nIter = iterations, LBFGS = True, u_f = lambd)

    # ****************************** Evaluating the model ******************************
    
    # Predict over whole data
    model.evaluate()

    # ****************************** Saving results ******************************
    
    print(f'\n\n\t\tSaving Results\n\t\t\t.')
    plot_ground_truth(X_star, u_star, SAVE_DIR)
    loss_plot(model, SAVE_DIR)
    plot_final_prediction(model, X_star, u_star, SAVE_DIR)
    print(f'\t\t\t.')
    save_gif_PIL(os.path.join(SAVE_DIR,"learning.gif"), model.files, fps=10, loop=0)
    print(f'\t\t\t.')