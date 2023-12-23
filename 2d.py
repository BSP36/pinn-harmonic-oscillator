import torch
import matplotlib.pyplot as plt
from model import WaveFunction

mass = 1.0
hbar = 1.0

kf = 0.5 * hbar ** 2 / mass

def train(psi, num_epoch, batch_size, optimizer, L_max, L_min, device="cpu"):
    psi = psi.to(device)
    L = L_max - L_min
    b = float(batch_size)

    for epoch in range(num_epoch):
        psi.train()
        optimizer.zero_grad()
        ham = 0.0
        norm = 0.0
        """
        Monte-Carlo integration or Newton–Cotes formulae
        """
        # x = [torch.rand(2, requires_grad=True) * L + L_min for _ in range(batch_size**2)]  
        x = []
        for i in range(batch_size):
            for j in range(batch_size):
                xy = torch.rand
                xy = torch.tensor([i/b, j/b], requires_grad=True).to(device) * L + L_min
                x.append(xy) 
        
        f = psi(torch.stack(x, dim=0))
        f = [f[i, 0].to(device) for i in range(batch_size ** 2)]
        """
        automatic differentiation
        """
        df = torch.autograd.grad(f, x, create_graph=True)
        for i in range(batch_size ** 2):
            f2 = f[i] ** 2
            kin = kf * (df[i][0] ** 2 + df[i][1] ** 2)
            pot = potential(x[i][0], x[i][1]) * f2
            ham += kin + pot
            norm += f2
        loss = ham / norm

        loss.backward()
        optimizer.step()

        print(epoch, loss.item())
        if (epoch+1) % 500 == 0:
            show_wavefunction(model=psi)



def show_wavefunction(model):
        model.eval()
        x = []
        N = 100
        L = 10.0
        dx = dy = L / N
        for i in range(N):
            for j in range(N):
                xy = torch.tensor([i/N, j/N], requires_grad=False) * L - 5.0
                x.append(xy) 
        x = torch.stack(x, dim=0)
        with torch.no_grad():
            y_out = model(x)
        norm = torch.sum(y_out ** 2) * dx * dy
        y_out /= torch.sqrt(norm)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:, 0], x[:, 1], y_out, c=y_out)  
        plt.show()


def potential(x, y):
    omega = 1.0
    return 0.5 * omega**2 * (x**2 + y**2)
            
            
if __name__ == "__main__":
    psi = WaveFunction(in_dim=2, num_mid_layers=5)
    optimizer = torch.optim.Adam(psi.parameters(), lr=1e-3)

    train(
        psi=psi,
        num_epoch=5000,
        batch_size=32,
        optimizer=optimizer,
        L_max=10.0,
        L_min=-10.0,
        device="cpu",
    )
