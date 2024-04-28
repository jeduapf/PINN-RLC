import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import matplotlib.pyplot as plt
import mat73
import os

def exact_solution(t, R, L, C, Vin, SNR = 0.0, mu = 0, sigma = 1, eps = 10**-9):
    assert isinstance(R,float) and isinstance(L,float) and isinstance(C,float) and isinstance(Vin,float), "All values R,L,C,Vin must be floats"
    assert R > 0 and L > 0 and C > 0 and Vin > 0, "All values R,L,C,Vin must be greater than 0"

    kappa = (C*R**2 - 4*L)/C

    if kappa < eps and kappa > 0:
        A = C*Vin
        alpha = -R/(2*L)
        sol = A*(1+np.exp(alpha*t)*(t-1))

    elif kappa > eps:
        A = C*Vin
        delta = np.emath.sqrt(kappa)

        alpha1 = (-R + delta)/(2*L)
        alpha2 = (-R - delta)/(2*L)
        sol = A*(1 +(alpha1/(alpha2-alpha1))*np.exp(alpha2*t) -(alpha2/(alpha2-alpha1))*np.exp(alpha1*t))

    else:
        A = C*Vin
        delta = np.emath.sqrt(kappa)

        alpha1 = (-R + delta)/(2*L)
        alpha2 = (-R - delta)/(2*L)
        sol = np.abs(A*(1 +(alpha1/(alpha2-alpha1))*np.exp(alpha2*t) -(alpha2/(alpha2-alpha1))*np.exp(alpha1*t)))

    # Adding noise 
    if SNR < 10**3:
        Vn = A/(10**(SNR/20))
    else :
        Vn = 0

    return sol[:,np.newaxis] + Vn*np.random.normal(mu, sigma, size=(len(t), 1))

def save_paths(folder = 'Results'):
    current_dir = os.getcwd()
    SAVE_DIR = os.path.join(current_dir,folder)
    try:
        os.mkdir(SAVE_DIR)
    except:
        pass

    SAVE_DIR_GIF = os.path.join(SAVE_DIR,'gif_imgs')
    try:
        os.mkdir(SAVE_DIR_GIF)
    except:
        pass

    return SAVE_DIR, SAVE_DIR_GIF

def import_matlab_data(out = "u.mat", inp = "sin.mat",DATA_PATH =r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-RLC"):
    u_star = mat73.loadmat(os.path.join(DATA_PATH, out))
    u_in_star = mat73.loadmat(os.path.join(DATA_PATH, inp))

    X_star = u_in_star['sin'].T
    u_star = u_star['u'][1,:].T
    u_star = u_star[:,np.newaxis]

    #          X_star       u_star
    #        [ t , v(t)]   [ i(t) ]
    #          t1 , v1        i1
    #          t2 , v2        i2
    # points     ...          ...
    #          tN , vN        iN
    return X_star, u_star

def plot_ground_truth(X_star, u_star, SAVE_DIR):
    fig= go.Figure(go.Scatter(x=X_star[:,0], y = X_star[:,1], mode ="lines", name="Input signal"))
    fig.add_scatter(x=X_star[:,0], y=u_star[:,0], mode='lines', name="Output signal")
    fig.write_html(os.path.join(SAVE_DIR, "ground_truth.html"))

def choose_points(N_u, X_star, u_star):  
    # create training set
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_train = X_star[idx,:]
    u_train = u_star[idx,:]

    return X_train, u_train, idx

def loss_plot(model, SAVE_DIR):
    loss = np.array(model.losses)
    X = np.linspace(0,loss.shape[0],loss.shape[0])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=( "Loss Data", "Loss f", "Total Loss"))

    fig.add_trace(
        go.Scatter(x=X, y=loss[:,0], name = "Loss Data"),
        row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x=X, y=loss[:,1], name = "Loss f"),
        row=2, col=1
        )

    fig.add_trace(
        go.Scatter(x=X, y=loss[:,2], name = "Total Loss"),
        row=3, col=1
        )
    fig.update_layout( height=1500, width=1500, 
                            title_text="Losses training", showlegend=False)

    fig.write_html(os.path.join(SAVE_DIR,"losses.html"))

def plot_final_prediction(model, X_star, u_star, SAVE_DIR):
    u_pred, _ = model.predict(X_star)
    X_obs = model.X_observations.detach().cpu().numpy()
    u_obs = model.u_observations.detach().cpu().numpy()
    initial_RL, initial_LC = model.guess

    res = np.abs(u_pred[:,0]-u_star[:,0])**2

    fig = make_subplots(    rows=2, cols=1,
                            shared_xaxes=True, 
                            subplot_titles=(
                            f"Prediction vs Ground Truth <br>Initial R/L = {initial_RL:.3f}, Expected R/L = {model.R/model.L:.3f}<br>Initial LC = {initial_LC:.3f}, Expected LC = {model.L*model.C:.3f}", 
                            f"<b>|u_pred-u_star|^2 (Sum = {np.sum(res):.3f} / Mean = {np.mean(res):.5f} )</b>"))
    fig.add_trace(
        go.Scatter( x=X_star[:,0], 
                    y=u_star[:,0], 
                    mode='lines', name = "Ground Truth",
                    line={
                            "color":"rgba( 0, 60, 250, 0.7)"
                        }),
        row=1, col=1
        )   
    fig.add_trace(
        go.Scatter( x=X_star[:,0], 
                    y=u_pred[:,0], 
                    mode='lines', name = "PINN prediction",
                    line={
                            "color":"rgba( 210, 0, 21, 0.8)"
                        }),
        row=1, col=1
        )  
    fig.add_trace(
        go.Scatter( x=X_obs[:,0], 
                    y=u_obs[:,0], 
                    mode='markers', name = "PINN data points",
                    marker={
                            "color":"rgba( 10, 10, 10, 0.45)",
                            'size': 4
                        }),
        row=1, col=1
        )   

    fig.add_trace(
        go.Scatter( x=X_star[:,0], 
                    y=res, 
                    mode='lines', name = "Residual = |u_pred-u_star|^2",
                    line={
                            "color":"rgba( 20, 20, 20, 0.9)"
                        }),
        row=2, col=1
        )   
    fig.add_hline(y=np.mean(res), line_dash="dot", name = "Mean of Squared Residual", row=2, col=1)
    fig.update_layout(  height=1200, width=1500, showlegend=True, 
                        title_text=f"Final full model prediction")
    fig.write_html(os.path.join(SAVE_DIR,"u_pred_vs_star.html"))

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def save_plt(i, model,  X_star, u_star):
    u_pred, _ = model.predict(X_star)
    X_obs = model.X_observations.detach().cpu().numpy()
    u_obs = model.u_observations.detach().cpu().numpy()
    res = np.abs(u_pred[:,0]-u_star[:,0])**2

    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(f"Observations and PINN approximation at iter: {i}")

    plt.subplot(2,1,1)
    plt.scatter(X_obs[:,0], u_obs[:,0], label="Observations", alpha=0.6)
    plt.plot(X_star[:,0], u_pred[:,0], label="PINN solution", color="tab:green")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(res, label=f"Residual^2 (Sum = {np.sum(res)})",color="red", alpha=0.7)
    plt.legend()

    return fig

def save_txt(string, PATH):
    with open(PATH, 'w') as file:
        file.write(string)