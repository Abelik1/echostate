import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from .Heisenberg_sim import HeisenbergChain
from qutip import Qobj, basis, tensor, sigmax, sigmay, sigmaz, qeye, expect
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))


class ESNPredictor:
    """
    Train an ESN on single-qubit reduced states extracted
    from array-based phistory of a HeisenbergChain.
    """
    def __init__(
        self,
        steps,
        dt,
        N,
        history_arrays: list,
        dims: list,
        qubit: int,
        hidden_dim=100,
        spectral_radius=0.9,
        input_scaling=1.0,
        ridge_param=1e-3,
        batch_size=1,
        model_path=None,
        leaky_rate = 0.9,
        sparsity = 1.0,
        training_depth = 1,
        washout = 0,
    ):
        # qubit_history: [⟨σ_z⟩_0, ⟨σ_z⟩_1, ..., ⟨σ_z⟩_T]
        
        
        self.steps = steps
        self.dt = dt
        self.N = N
        self.test_history = history_arrays
        self.dims = dims
        self.q = qubit
        self.model_path = model_path
        self.batch_size = batch_size
        self.hidden_dim=hidden_dim
        self.spectral_radius=spectral_radius
        self.input_scaling=input_scaling
        self.ridge_param= ridge_param
        self.leaky_rate = leaky_rate
        self.sparsity = sparsity
        self.training_depth = training_depth
        self.washout = washout
        
        self.history = []
        for _ in range(training_depth):
            chain = HeisenbergChain(self.N, self.dt)
            chain.evolve(steps=self.steps)
            self.history.append(chain.history)
        print("Saved all histories")
        
            
    def train(self):
        if self.q ==0:
            self.pbar = tqdm(total=(self.N*self.training_depth), desc="Processing qubits")
        else:
            self.pbar = tqdm(total=(self.training_depth), desc="Processing qubits")
            
        for i in range(self.training_depth):
            for qubit in (range(self.N) if self.q == 0 else [self.q]):
                self.esn._esn_cell.hidden.fill_(0.0) 
                
                qubit_history = [float(expect(sigmaz(), Qobj(rho, dims=self.dims).ptrace(qubit))) for rho in self.history[i]]
                # print(qubit_history[:9])
                
                # Create autoregressive input: [x_t, y_{t-1}]
                X_vals = [[qubit_history[t], qubit_history[t-1]] for t in range(1, len(qubit_history)-1)]
                Y_vals = [qubit_history[t] for t in range(2, len(qubit_history))]
                # print(X_vals[:10])
                X = torch.tensor(X_vals, dtype=torch.float32).view(-1,1,2) # Can change this 2 to 1 if no Feedback
                Y = torch.tensor(Y_vals, dtype=torch.float32).view(-1,1,1)
                
                self.X = X.to(device)
                self.Y = Y.to(device)
                
                loader = DataLoader(TensorDataset(self.X, self.Y), batch_size=self.batch_size, shuffle=False)
                for xb, yb in loader:
                    self.esn(xb, yb)
                self.pbar.update(1)
                # print(self.esn._esn_cell.w)
   
            
        self.esn.output.xTx = self.esn.output.xTx.cpu()
        self.esn.output.xTy = self.esn.output.xTy.cpu()         
        self.esn.finalize()               # closes ridge solution
        self.esn.output.w_out = self.esn.output.w_out.to(self.X.device)
        self.pbar.close()
        
        
    def run(self):
        # train or load existing model
        if self.model_path and os.path.exists(self.model_path):
            
            
            # self.esn.load_state_dict(torch.load(self.model_path))
            self.esn = torch.load(self.model_path)
                         
            self.esn.eval()
        else:
            # instantiate ESN (EchoTorch v1.8.1 signature)
            self.esn = LiESN(
                    input_dim=2, # Change if no Feedback
                    hidden_dim=self.hidden_dim,
                    output_dim=1,
                    spectral_radius=self.spectral_radius,
                    input_scaling=self.input_scaling,
                    ridge_param=self.ridge_param,
                    leak_rate=self.leaky_rate,
                    learning_algo="inv",
                    washout = self.washout,

                ).to(device)
            
            self.train()
            
            torch.save(self.esn, self.model_path)
            # torch.save(self.esn.state_dict(), self.model_path)
            print(f"✅ Saved trained model to {self.model_path}")
        
                
    def predict(self,):  
        qubit_history = [float(expect(sigmaz(), Qobj(rho, dims=self.dims).ptrace(self.q))) for rho in self.test_history]
        X_vals = [[qubit_history[t], qubit_history[t-1]] for t in range(1, len(qubit_history)-1)]
        X = torch.tensor(X_vals, dtype=torch.float32).view(-1,1,2) # Can change this 2 to 1 if no Feedback
        self.X = X.to(device)
        # self.debug_singular_matrix()      # still valid
        
        self.predictions = []
        prev_out = self.X[0, 0, 1].item()  # initial previous output

        with torch.no_grad():
            for t in range(len(self.X)):
                if t < self.washout:
                    curr_in = self.X[t, 0, 0].item()
                    input_tensor = torch.tensor([[ [curr_in, prev_out] ]], dtype=torch.float32).to(device)  # shape [1,1,2]
                    prev_out = curr_in
                    
                    num = self.esn(input_tensor)
                    print("Num ",num)
                    
                if t>=self.washout:
                    curr_in = self.X[t, 0, 0].item()
                    input_tensor = torch.tensor([[ [curr_in, prev_out] ]], dtype=torch.float32).to(device)  # shape [1,1,2]
                    # print("Input tenstor: ", input_tensor)
                    num = self.esn(input_tensor)
                    # print("Num ",num)
                    
                    out = num.squeeze().cpu().item()
                    
                    prev_out = out  # feedback
                    self.predictions.append(out)
        
    def objective(self, trial):
        """
        Optuna objective: train ESN on a fresh trial config and compute mean abs error.
        """
        # Suggest hyperparameters
        hidden_dim      = trial.suggest_int   ("hidden_dim",      200, 1000)
        spectral_radius = trial.suggest_float ("spectral_radius", 0.1, 1.4)
        input_scaling   = trial.suggest_float ("input_scaling",   0.05, 5.0)
        ridge_param     = trial.suggest_float ("ridge_param",     1e-7, 1.0, log=True)
        leaky_rate      = trial.suggest_float ("leaky_rate",      0.1, 1.0)
        sparsity = trial.suggest_float ("sparsity",      0.05, 1.0)
        # Build ESN
        self.esn = ESN(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=1,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            ridge_param=ridge_param,
            leaky_rate=leaky_rate,
            learning_algo="inv",
           
        ).to(device)

        self.train()

        # Predict with feedback matching your predict() logic
        self.predict()

        preds = np.array(self.predictions)
        true_vals = np.array([y.item() for y in self.Y[self.washout:].cpu().view(-1)])

        # Compute mean absolute error
        return float(np.mean(np.abs(preds - true_vals)))

    def debug_singular_matrix(self):
        """
        Inspect the accumulated covariance matrix in the RRCell to diagnose singularity.
        """
        # After calling esn(X, Y), the RRCell has accumulated xTx and xTy
        rc = self.esn.output
        cov = rc.xTx
        try:
            cov_arr = cov.toarray()
        except AttributeError:
            cov_arr = cov
        import numpy.linalg as LA
        rank = LA.matrix_rank(cov_arr.cpu().numpy())
        cond = LA.cond(cov_arr.cpu().numpy())
        print(f"[DEBUG] cov shape: {cov_arr.shape}")
        print(f"[DEBUG] Rank(cov): {rank}/{cov_arr.shape[0]}")
        print(f"[DEBUG] Condition number: {cond:.2e}")

import re

if __name__ == '__main__':
    steps = 5000
    N = 3
    dt = 0.01
    qubit = 1 # Uses index notation so be careful
    
    if qubit >= N:
        raise IndexError("You have selected a qubit that doesn't exist")
        
    
    name = f"Stps{int(steps/1000.0)}_Qbts{N}_dt{dt}".replace(".","_",1) #split("_")[1].rsplit(".",1)[0][3:]
    path = f'./cache/history_data_{name}.pickle'
    chain = HeisenbergChain(N, dt)
    # simulate chain with array storage
    try:
        with open(path, 'rb') as f:
            chain.history = pickle.load(f)
        print("Found a data file: ",path.split("_",1)[1].rsplit(".",1)[0])
    except:  
        chain.evolve(steps=steps)
        with open(path, 'wb') as f:
            pickle.dump(chain.history, f)
            
    # chain.plot_spin_grid(size) # Size is too large to graph it anymore
    # stats = chain.check_conservation()
    # print('Conservation:', stats)
    # plt.show()

    predictor = ESNPredictor(
        steps,
        dt,
        N,
        history_arrays=chain.history,
        dims=chain.dims,
        qubit=qubit,
        hidden_dim=208,
        spectral_radius=0.3935,
        ridge_param=3.368e-5,
        leaky_rate=0.251, 
        input_scaling=4.7847,
        sparsity = 0.1,
        # feedback=False,#  Done manually
        model_path=f"./cache/trained_esn_{name}.pt",
        training_depth = 4,
        washout = 0
    )
    
    
    # study = optuna.create_study(direction="minimize")
    # study.optimize(predictor.objective, n_trials=100)
    # print("Best trial:", study.best_trial.params)


    
    # fidelity plot
    predictor.run()
    predictor.predict()
    
    true_rhos = [Qobj(arr, dims=chain.dims).ptrace(1) for arr in chain.history[1:]][1000:1400:1]
    # print(predictor.predictions)
    preds = [p[0] if isinstance(p, list) else p for p in predictor.predictions][1000:1400:1]
    true_vals = [float(expect(sigmaz(), rho)) for rho in true_rhos]
    errors = [abs(p - t) for p, t in zip(preds, true_vals)]
    
    plt.figure()
    plt.plot(errors, marker='x')
    plt.title("Prediction Error |⟨σ_z⟩_pred - ⟨σ_z⟩_true|")
    
    plt.savefig(re.sub(r"(Qbts)", r"\1({})".format(qubit+1), f"cache/Errors_{name}.png"))
    
    plt.figure()
    plt.plot(preds,label = "Pred Value")
    plt.plot(true_vals, label = "True Value")
    plt.legend()
    
    plt.show()
