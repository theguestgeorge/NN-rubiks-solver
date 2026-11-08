import model
import scramble
from torch import nn, torch

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return torch.relu(x + self.net(x))

class PolicyNet(nn.Module):
    def __init__(self, hidden=1024, num_blocks=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(480, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(num_blocks)])
        self.out = nn.Linear(hidden, 18)
    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))