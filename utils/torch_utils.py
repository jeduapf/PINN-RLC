import torch
from torchsummary import summary
from collections import OrderedDict
import numpy as np
from utils import *

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ANN(torch.nn.Module):
    '''
        Defining the architecture of the Neural Network.
        According to https://github.com/maziarraissi/PINNs.

        Ex.: F(t,a,b,c,d,e) = (g(t,a,b,c,d,e), h(t,a,b,c,d,e), v(t,a,b,c,d,e))

        So there would be 6 neurons in the input layer and 3 neurons as the output layer
        Since there are 6 entries (t,a,b,c,d,e) and 3 output functions (g,h,v)

        Thus layers = [6,20,20,20,20,20,20,20,20,3]

        Input:
            - inputs[int]: First layer neurons => Input Space of the function
            - outputs[int]: Last layer neurons => Output Space of the function
            - activation_func[Torch.nn]: Any activation function from torch as in https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity. Standard nn.Tanh

        Output:
            - torch object model. Linked Sequential layers

    '''
    def __init__(self, inputs, outputs, activation_func = torch.nn.Tanh):

        assert isinstance(inputs,int) and isinstance(outputs,int) and inputs > 0 and outputs > 0, "Input layer and Output layer must be non negative values except 0 !"
        
        # Inherit pytorch nn class
        super(ANN, self).__init__()

        self.inp = inputs
        self.out = outputs
        self.ann = [self.inp, 20, 20, 20, 20, 20, 20, 20, 20, self.out]
        self.depth = len(self.ann) - 1
        self.activation = activation_func

        layer_dict = {}
        for i in range(self.depth - 1): 
            layer_dict[f'layer_{i}'] = torch.nn.Linear(self.ann[i], self.ann[i+1])
            layer_dict[f'activation_{i}'] = self.activation()
        
        layer_dict[f'layer_{(self.depth - 1)}'] = torch.nn.Linear(self.ann[-2], self.ann[-1])
        
        # deploy layers
        self.layers = torch.nn.Sequential(OrderedDict(layer_dict)).to(device)


    # For prediciton 
    def forward(self, x):
        return self.layers(x)

# As in https://github.com/maziarraissi/PINNs
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(1.0)

class PhysicsInformedNN:
    # Initialize the class

    '''
        X is the matrix type object that contains the measurements of the input variables of the wanted function.
        For example: 
            given the wanted equation h(x,y,z,t) and its differential equation:
                        a_1*h_tt + a_2*h_xy + a_3*h_yy + a_4*h_yz = 0

        X will be of the following shape:
                                             0 , 1 , 2 , 3 
                        X[t, x, y, z] = [   [t1, x1, y1, z1 ],
                                            [t2, x2, y2, z2 ],
                                                   ...     
                                            [tn, xn, yn, zn ]]

        So to get y input values of the function h, just cut the matrix as X[:,2] => [y1, y2,..., yn]
                                                                           X[:,2:3] => [[y1, y2,..., yn]]
    '''
    def __init__(self, X_star, u_star, X_observations, u_observations, R ,L ,C, inputs, outputs, guess, SHOW_MODEL, SHOW_PRINTS= False, SHOW_ITER = 100, GIF_FIGS = 99, SAVE_DIR_GIF = None):
        
        # Visualizations variables
        self.SHOW_ITER = SHOW_ITER
        self.SHOW_PRINTS = SHOW_PRINTS
        self.GIF_FIGS = GIF_FIGS
        self.SAVE_DIR_GIF = SAVE_DIR_GIF
        self.files = []

        # Ground Truth
        self.X_star = X_star
        self.u_star = u_star
        self.R = R
        self.L = L
        self.C = C

        # boundary conditions
        self.lb = torch.tensor(self.X_star.min(0)).float().to(device) # Lower bound
        self.ub = torch.tensor(self.X_star.max(0)).float().to(device) # Upper bound
        
        # data
        self.t = torch.tensor(X_observations[:, 0:1], requires_grad=True).float().to(device)
        self.u_in = torch.tensor(X_observations[:, 1:2], requires_grad=True).float().to(device)
        self.u_observations = torch.tensor(u_observations).float().to(device)
        self.X_observations =  torch.tensor(X_observations).float().to(device)

        # Parameters
        self.guess = guess
        self.RL = torch.nn.Parameter(torch.tensor([float(self.guess[0])], requires_grad=True).to(device))
        self.LC = torch.nn.Parameter(torch.tensor([float(self.guess[1])], requires_grad=True).to(device))
        
        # deep neural networks
        self.ann = ANN(inputs, outputs, activation_func = torch.nn.Tanh).to(device)
        self.ann.register_parameter('RL', self.RL)
        self.ann.register_parameter('LC', self.LC)
        self.ann.apply(init_weights)

        if SHOW_MODEL:
            print("\n\n---------------- MODEL ARCHITECTURE----------------\n")
            summary(self.ann,(1,inputs))
            print("\n\n")
        
        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.ann.parameters(), 
            lr=1.0, 
            max_iter=5000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
        
        self.optimizer_Adam = torch.optim.Adam(self.ann.parameters(), lr = 0.01)
        # self.optimizer_Adam = torch.optim.SGD(self.ann.parameters(), lr=0.1, momentum=0.9)
        self.iter = 0
        self.losses = []

    def net_u(self, t):  
        u_pinn = self.ann(torch.cat([t], dim=1))
        return u_pinn
    
    def net_f(self, t, u_in):
        """ The pytorch autograd version of calculating residual """
        RL = self.RL        
        LC = self.LC
        u_pinn = self.net_u(t)

        u_t = torch.autograd.grad(
            u_pinn, t, 
            grad_outputs=torch.ones_like(u_pinn),
            retain_graph=True,
            create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t, 
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True
        )[0]


        # u_tt = -( R/L )*u_t -( 1/(L*C) )*u + ( 1/(L*C) )*u_in
        f = u_tt + RL*u_t + LC*u_pinn - LC*u_in
        
        return f
    
    def loss_func(self):
        u_pred = self.net_u(self.t)
        f_pred = self.net_f(self.t, self.u_in)
        loss_u = torch.mean((self.u_observations - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f
        self.losses.append([loss_u.item(),
                    loss_f.item(),
                    loss.item()])

        self.optimizer.zero_grad()
        loss.backward()
        if self.GIF_FIGS is not None:
            if self.iter % self.GIF_FIGS == 0:
                fig = save_plt(self.iter, self,  self.X_star, self.u_star)
                file = os.path.join(self.SAVE_DIR_GIF,"pinn_%.8i.png"%(self.nIter + self.iter+1))
                fig.savefig(file, dpi=100, facecolor="white")
                self.files.append(file)
                plt.close(fig)

        self.iter += 1
        if self.SHOW_PRINTS:
            if self.iter % self.SHOW_ITER == 0:
                print(
                    'Iter: %d, Loss: %.3e, Loss_u: %.3e, Loss_f: %.3e, R/L: %.3f, L*C: %.6f' % 
                    (
                        self.iter,
                        loss.item(),
                        loss_u.item(),
                        loss_f.item(), 
                        self.RL.item(), 
                        self.LC.item()
                    )
                )
        return loss
    
    def train(self, nIter, LBFGS, u_f = 10**0):

        self.nIter = nIter
        # Setting the model in training mode
        self.ann.train()

        if self.SHOW_PRINTS: 
            print(f"\nInput variables: \n\tR: {self.R}\n\tL: {self.L}\n\tC: {self.C}\n")
            print(f"\nGuess R/L: {self.guess[0]}\t\t Guess L*C: {self.guess[1]}\n\n")

            print(f'\n\n\t\tSTARTING ADAM !\n\n')

        for epoch in range(nIter):
            u_pred = self.net_u(self.t)
            f_pred = self.net_f(self.t, self.u_in)
            loss_u = torch.mean((self.u_observations - u_pred) ** 2)
            loss_f = torch.mean(f_pred ** 2)

            # Calculating total loss
            loss = u_f*loss_u + loss_f
            self.losses.append([loss_u.item(),
                                loss_f.item(),
                                loss.item()])

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            
            if self.GIF_FIGS is not None:
                if epoch % self.GIF_FIGS == 0:
                    fig = save_plt(epoch, self,  self.X_star, self.u_star)
                    file = os.path.join(self.SAVE_DIR_GIF,"pinn_%.8i.png"%(epoch+1))
                    fig.savefig(file, dpi=100, facecolor="white")
                    self.files.append(file)
                    plt.close(fig)

            if self.SHOW_PRINTS:
                if epoch % self.SHOW_ITER == 0:
                    print(
                        'It: %d, Loss: %.3e, Loss_u: %.3e, Loss_f: %.3e, R/L: %.3f, L*C: %.6f' % 
                        (
                            epoch, 
                            loss.item(),
                            loss_u.item(), 
                            loss_f.item(), 
                            self.RL.item(), 
                            self.LC.item()
                        )
                    )
        if self.SHOW_PRINTS:
            print(f'\n\n\t\tSTARTING FINE TUNE!\n\n')

        if LBFGS : 
            self.optimizer.step(self.loss_func)
    
    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        u_in = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.ann.eval()
        u_pred = self.net_u(t)
        f = self.net_f(t, u_in)
        u_pred = u_pred.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u_pred, f

    def evaluate(self):
        # Predict over whole data
        u_pred, f_pred = self.predict(self.X_star)
        error_u = np.linalg.norm(self.u_star-u_pred,2)/np.linalg.norm(self.u_star,2)

        RL = self.RL.detach().cpu().numpy()
        LC = 1/self.LC.detach().cpu().numpy()

        error_lambda_1 = 100*np.abs(RL - self.R/self.L) / (self.R/self.L) 
        error_lambda_2 = 100*np.abs(LC - self.L*self.C) / (self.L*self.C) 

        if self.SHOW_PRINTS:
            print(f'\n\n\t\tMODEL ALL DATA EVALUATION\n\n')
            print('Error u: %e' % (error_u))
            print(f'R/L: {RL[0]}')
            print('Error R/L: %.5f%%' % (error_lambda_1[0]))
            print(f'L*C: {LC[0]}')
            print('Error L*C: %.5f%%' % (error_lambda_2[0]))
