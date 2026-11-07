import random as rand
from cube import *
from model import encode
import torch

MOVE_NAMES = ["R", "R'", "R2", 
             "L", "L'", "L2",
             "U", "U'", "U2",
             "D", "D'", "D2",
             "F", "F'", "F2",
             "B", "B'", "B2"
]

#used for generating a single random scramble 
def random_scramble(length, rng=rand):
    moves, last_face = [], None
    for _ in range(length):
        #make sure after a move there isnt a reducable move for example
        #moving R then R2 is the equivalent as R' and we dont want to pollute the labelled data
        #so we remove any variations of R from the list of moves to be chosen from
        m = rng.choice([m for m in MOVE_NAMES if m[0] != last_face])
        moves.append(m)
        last_face = m[0]
    return moves


#creates a x amount of scrambles, for each scramble it records the state
#and the move that gets u y amount closer to the solved state 
def make_samples(num_scrambles, length=20):
    xs, ys = [], []
    for _ in range(num_scrambles):
        state = SOLVED_CUBE
        for move in random_scramble(length):
            state = state.apply_move(move)
            xs.append(encode(state))
            ys.append(MOVE_NAMES.index(invert_move(move)))  # label = undo move
    return torch.stack(xs), torch.tensor(ys)

