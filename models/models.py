import torch
from torch import nn
from models.double_sided_crystal_ball import DoubleSidedCrystalBall,DoubleSidedCrystalBall_EGM

class RegressionDNN(nn.Module):
    def __init__(self, features, hidden_net=[32,32], boundary=[-5,5]): 
        super().__init__()
        self.features = features
        self.boundary = boundary
        in_dim = self.features
        # DNN for params (shift is fixed)
        layers = []
        self.features = features
        for i in range(len(hidden_net)):
            layers.append(nn.Linear(in_dim, hidden_net[i]))
            layers.append(nn.ReLU())  # You can change the activation function
            layers.append(nn.LayerNorm(hidden_net[i]))
            in_dim = hidden_net[i]
        layers.append(nn.Linear(in_dim, 6))  # output for scale
        self.hyper = nn.Sequential(*layers)
        self.mu_scale = torch.as_tensor(boundary[1]-boundary[0])
        self.mu_min = torch.as_tensor(boundary[0])

    def forward(self, x): # Add forward method
        out = self.hyper(x)
        # apply a sigmoid because all the parameters apart from the mu needs to be positive
        mu = self.mu_min + self.mu_scale* torch.sigmoid(out[:,0])
        params = torch.nn.functional.softplus(out[:,1:]) + 1e-8
        return DoubleSidedCrystalBall(mu=mu,
                                      width=params[:,0],
                                      a1=params[:,1],
                                      a2=params[:,2],
                                      p1=params[:,3],
                                      p2=params[:,4], 
                                     xmin=self.boundary[0],
                                     xmax=self.boundary[1])

    def rsample(self, x, sample_shape, n_samples_cdf_inversion=40):
        d = self.forward(x) # Call forward method
        return d.rsample(sample_shape, n_samples_cdf_inversion)

    def log_prob(self, value, x):
        d = self.forward(x) # Call forward method
        return d.log_prob(value)


class RegressionDNN_EGM(nn.Module):
    def __init__(self, features, hidden_net=[32,32], boundary=[-5,5], alphaL_min=1, alphaR_min=1,fix_mu=False): 
        super().__init__()
        self.features = features
        self.boundary = boundary
        self.alphaL_min = alphaL_min
        self.alphaR_min = alphaR_min
        self.fix_mu = fix_mu
        in_dim = self.features
        # DNN for params (shift is fixed)
        layers = []
        self.features = features
        for i in range(len(hidden_net)):
            layers.append(nn.Linear(in_dim, hidden_net[i]))
            layers.append(nn.GELU())  # You can change the activation function
            layers.append(nn.LayerNorm(hidden_net[i]))
            in_dim = hidden_net[i]
        layers.append(nn.Linear(in_dim, 6))  # output for scale
        self.hyper = nn.Sequential(*layers)
        

    def forward(self, x): # Add forward method
        out = self.hyper(x)
        # apply a sigmoid because all the parameters apart from the mu needs to be positive

        if not self.fix_mu:
            mu = torch.log(torch.tensor(2))*(torch.sigmoid(out[:,0])-0.5)
        else:
            # In the second step, the fraction E_raw/E_true is fixed to 1. As our target is log(E_raw/E_true), we set mu to 0
            mu = torch.zeros_like(out[:,0])  # mu is fixed to 0
        params = torch.nn.functional.softplus(out[:,1:])
        return DoubleSidedCrystalBall_EGM(mu=mu.unsqueeze(-1),
                                      sigma=params[:,0].unsqueeze(-1),
                                      alphaL=params[:,1].unsqueeze(-1)+self.alphaL_min,
                                      alphaR=params[:,2].unsqueeze(-1)+self.alphaR_min,
                                      etaL=params[:,3].unsqueeze(-1)+1,
                                      etaR=params[:,4].unsqueeze(-1)+1)

    def rsample(self, x, sample_shape, n_samples_cdf_inversion=40):
        d = self.forward(x) # Call forward method
        return d.rsample(sample_shape, n_samples_cdf_inversion)

    def log_prob(self, value, x):
        d = self.forward(x) # Call forward method
        return d.log_prob(value)




