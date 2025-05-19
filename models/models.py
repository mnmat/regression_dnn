import torch
from torch import nn
from models.double_sided_crystal_ball import DoubleSidedCrystalBall

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

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

