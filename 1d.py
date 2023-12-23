import torch
import matplotlib.pyplot as plt
from model import WaveFunction

mass = 1.0
hbar = 1.0

kf = -0.5 * hbar ** 2 / mass


def train(psi, num_epoch, batch_size, optimizer, L_max, L_min, device="cpu"):
    psi = psi.to(device)
    L = L_max - L_min

    for epoch in range(num_epoch):
        psi.train()
        optimizer.zero_grad()
        ham = 0.0
        norm = 0.0
        """
        Monte-Carlo integration or Newton–Cotes formulae
        """
        # x = [torch.rand(1, requires_grad=True) * dL + L_min for _ in range(batch_size)]    
        x = [torch.tensor([i/float(batch_size)], requires_grad=True).to(device) * L + L_min for i in range(batch_size)]  
        y = psi(torch.stack(x, dim=0))
        y = [y[i, 0].to(device) for i in range(batch_size)]
        """
        automatic differentiation
        """
        dy = torch.autograd.grad(y, x, create_graph=True)
        ddy = torch.autograd.grad(dy, x, create_graph=True)

        for i in range(batch_size):
            ham += y[i] * (potential(x[i]) * y[i] + kf * ddy[i])
            norm += y[i] * y[i]
        loss = ham / norm

        loss.backward()
        optimizer.step()

        print(epoch, loss.item())
        if (epoch+1) % 500 == 0:
            show_wavefunction(
                model=psi,
                x_in=torch.arange(start=-10.0, end=10.0, step=0.1),
                dx=L/batch_size,
            )

def potential(x):
     omega = 1.0
     return 0.5 * (omega * x) ** 2

def show_wavefunction(model, x_in, dx):
        model.eval()
        with torch.no_grad():
            y_out = model(x_in[:, None])
        norm = torch.sum(y_out ** 2) * dx
        
        plt.scatter(x_in[:], y_out[:, 0] / torch.sqrt(norm), s=1)  
        plt.show()


if __name__ == "__main__":
    psi = WaveFunction(in_dim=1, num_mid_layers=5)
    optimizer = torch.optim.Adam(psi.parameters(), lr=1e-3)

    train(
        psi=psi,
        num_epoch=5000,
        batch_size=512,
        optimizer=optimizer,
        L_max=15.0,
        L_min=-15.0,
        device="cpu",
    )
