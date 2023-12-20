import torch.nn as nn


class WaveFunction(nn.Module):
    def __init__(self, in_dim, num_mid_layers):
        super(WaveFunction, self).__init__()
        mid_dim = 32
        self.lr = nn.ModuleList()
        self.lr.append(
            nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.SiLU(),
            )
        )
        for _ in range(num_mid_layers):
            self.lr.append(
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.SiLU(),
                )
            )
        self.lr.append(nn.Sequential(nn.Linear(mid_dim, 1)))
    
    def forward(self, x):
        for layer in self.lr:
            x = layer(x)
        return x
