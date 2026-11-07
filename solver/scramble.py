import random as rand

def generate_scramble():
    moves = ["R", "R'", "R2", 
             "L", "L'", "L2",
             "U", "U'", "U2",
             "D", "D'", "D2",
             "F", "F'", "F2",
             "B", "B'", "B2"
    ]

    scramble = []
    for i in range(21):
        scramble.append(rand.choice(moves))

    print(scramble)

