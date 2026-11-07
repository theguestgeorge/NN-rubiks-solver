from cube import *
from torch import * 

#this encodes the state of the cube as a vector so the neural network
#can perfrom operations on it 
def encode(state) -> torch.Tensor:
    x = torch.zeros(20, 24)
    for slot, (piece, ori) in enumerate(zip(state.corners, state.corner_orientations)):
        x[slot, piece * 3 + ori] = 1
    for slot, (piece, ori) in enumerate(zip(state.edges, state.edge_orientations)):
        x[8 + slot, piece * 2 + ori] = 1
    return x.flatten()  # shape (480,)

def ResBlock():
    pass

def PolicyNet():
    pass