#Richard Tran
#CS362
#Spring 2019

# Imports
import random
import math

# (20^2 + 160 )  * 28 * 1600 * 64^3 for one gradient
#                      ^ size of compute activation
# Compute Activations: 2*[4*20] + [20 * 20 ] = 200
# Compute Gradient: (2 * Compute Activation  + 4) * Compute Activation = 404 * 200 = 80800
# 1 Generation of Training: Compute Gradient * 2^18

# Constants
MAX_PILE_SIZE = 64
LEARNING_RATE = 0.00001
GRADIENT_CHANGE = 0.000001
TOTAL_GENERATIONS = 5 # This needs to be short, since it takes so long per generation

# The neural net itself.
# Layer 1 is [which_turn. (0 for Player 1, 1 for Player 2)
#             pile_size1,
#             pile_size2,
#             pile_size3]
# Layer 2 and 3 are arbitrary
# Layer 4 is [pile1_activation,
#             pile2_activation,
#             pile3_activation,
#             new_pile_size]. The one with the highest activation gets subtracted from,
#                             assuming it is a valid move.
level_size = [4, 2, 2, 4] # NOTE: The hidden layers are this low because it takes a long time to calculate
                          #        all possible solutions

# Used to indicate which player goes first.
# 0 - Human
# 1 - CPU
first_player = 0

# Initialize edge weights by looping through all elements in the triply indexed array
# and assigns random value between 0 and 1.
# weights[i][j][k] is the edge weight from the j'th neuron in level i to the k'th neuron in level i+1.
weights = [[[random.uniform(0, 1) for k in range(level_size[i + 1])]
 for j in range(level_size[i])]
  for i in range(len(level_size) - 1)]

activations = [[1 for i in range(level_size[i])] for i in range (len(level_size))]

# Memoized list
solutions_memo = {}

# Retrieves the proper solution given a game state.
def get_solution_from_game_state(game_state):
    return solutions_memo.get(game_state)

# Just a ReLU function.
def ReLU( z ):
    return max(0,z)

# This function is a minimally modified version of the compute_activations function given by Professor Hayes.
# Steps through all but the first layer and computes activations.
# Forward computation.
def compute_new_activations():
    for i in range(1, len(level_size)):
        for k in range(level_size[i]):
            z = 0.0
            for j in range(level_size[i - 1]):
                z += activations[i - 1][j] * weights[i - 1][j][k]
            activations[i][k] = ReLU(z)
    activations[3][3] = math.floor(activations[3][3])
    return activations


# Creates a dictionary that maps game state with a proper solution to that game state.
# The game state holds pile sizes for all three piles with the key being which pile to choose, and its new size.
def find_all_solutions():
    for i in range(MAX_PILE_SIZE):
        for j in range(MAX_PILE_SIZE):
            for k in range(MAX_PILE_SIZE):
                game_state = (i, j, k)
                if game_state not in solutions_memo.keys():
                    subtraction = i ^ j ^ k
                    if i > j:
                        if i > k:
                            choice = 1
                            pile_value = i
                        else:
                            choice = 3
                            pile_value = k
                    elif j > k:
                        choice = 2
                        pile_value = j
                    else:
                        choice = 3
                        pile_value = k
                    new_pile_value = pile_value - subtraction
                    solution = (choice, new_pile_value)
                    solutions_memo.update({game_state : solution})


# Computes the gradient of the cost function.
# 
# 1. Compute gradient at top level (for square loss function).
# 2. Loop.  Update gradient with respect to weights first, then activations at previous level.
# 3. Return (gradient w.r.t. weights)
def compute_gradient(game_state):
    grad = [[[0 for w in v] for v in u] for u in weights]
    correct_value = get_solution_from_game_state(game_state)
    correct_pile = correct_value[0]
    correct_size = correct_value[1]
    for k in range(4):
        if correct_pile == k:
            value = 1
        elif k == 3:
            value = correct_size
        else:
            value = 0
        for i in range(len(activations)-1):
            for j in range(len(activations[i])):
                for l in range(len(activations[i+1])):
                    a = compute_new_activations( )
                    cost = (a[-1][k] - value) ** 2
                    #print(i, " ",j, " ", k)
                    #print (weights[1][1][1])
                    weights[i][j][l] = weights[i][j][l] + GRADIENT_CHANGE
                    changed_a = compute_new_activations( )
                    changed_cost = (changed_a[-1][0] - value) ** 2
                    weights[i][j][l] = weights[i][j][l] - GRADIENT_CHANGE
                    grad[i][j][l] = (changed_cost - cost)/GRADIENT_CHANGE 

    return grad

# This function is a minimally modified function given by Professor Hayes.
def train_net():
    sum = [ [ [0 for k in range(level_size[i+1])] for j in range(level_size[i])] for i in range(len(level_size)-1)]
    for pos in solutions_memo:
        g = compute_gradient( pos )
        # For every single position, get the gradient and sum it up.
        for i in range(len(level_size)-1):
            for j in range(level_size[i]):
                for k in range(level_size[i+1]):
                    sum[i][j][k] += g[i][j][k]
    for i in range(len(level_size)-1):
        for j in range(level_size[i]):
            for k in range(level_size[i+1]):
                weights[i][j][k] = weights[i][j][k] - (sum[i][j][k] / len(solutions_memo)) * LEARNING_RATE
    return activations

# Ask the user how many coins/rocks/sticks they would like to start for each pile.
def get_inputs ():
    activations[0][1] = int(input("How many pieces for the first pile? : "))
    activations[0][2] = int(input("How many pieces for the second pile? : "))
    activations[0][3] = int(input("How many pieces for the third pile? : "))
    activations[0][0] = first_player


def is_game_over () :
    total_piece = 0
    for i in range(1,4):
        total_piece += activations[0][i]
    if total_piece > 0:
        return 0
    else:
        return 1


def print_game():
    for i in range(1,4):
        print("Pile ", i, " : ", activations[0][i])

def find_average_cost( ):
    total = 0.0
    count = 0
    #For every solution, solve it and get the cost of each neuron and add that to the cost. Divide by amount of solutions.
    for solution in solutions_memo:
        for i in range(4):
            if i == 3:
                value = solution[1]
            else:
                value = solution[0]

            a = compute_new_activations( )
            cost = (a[-1][i] - value)**2
            total += cost
        count += 1
    return total / count

def play_game():
    get_inputs()
    while not is_game_over():
        if activations[0][0] == 0:
            which_pile = int(input("Select pile : "))
            subtraction = int(input("How many to take? : "))
            if subtraction > activations[0][which_pile]:
                print("Pile ", which_pile, " invalid choice. Not enough pieces.")
            else:
                previous_value = activations[0][which_pile]
                activations[0][which_pile] = previous_value - subtraction
                activations[0][0] = 1
        if activations[0][0] == 1:
            compute_new_activations()
            pile_1_activation = activations[3][0]
            pile_2_activation = activations[3][1]
            pile_3_activation = activations[3][2]
            new_pile_size = activations[3][3]
            if pile_1_activation > pile_2_activation:
                if pile_1_activation > pile_3_activation:
                    choice = 1
                else:
                    choice = 3
            elif pile_2_activation > pile_3_activation:
                choice = 2
            else:
                choice = 3
            print("Computer changes pile ", choice, " to ", new_pile_size)
            activations[0][choice] = new_pile_size
            activations[0][0] = 0

avg_costs = [ ]
print("Training Neural Net... (This may take up to 15 minutes with 4x2x2x4 Neural Net with i7-8650U @ 1.90 GHz...")
find_all_solutions()
for i in range(TOTAL_GENERATIONS):
    print("Generation: ", i, " Average Cost: ", find_average_cost( ))
    x = find_average_cost( )
    avg_costs.append(x)
    train_net()
print("Training complete! Let's play!")
import matplotlib.pyplot as plt

plt.scatter(range(TOTAL_GENERATIONS), avg_costs, color='darkgreen', marker='^')
plt.xlim(0.5, 4.5)
plt.show()

play_game()