import random
import copy
import time
import sys




start = time.time()

f = open("input.txt")

f2 = open("output.txt", "w+")

f1 = f.readlines()

state_of_player = int(f1[0].strip())

previous_board = []
current_board = []
num_tiles = 5

for i in range (1,6):
    previous = f1[i].strip()
    prlist = [int(d) for d in previous]
    previous_board.append(prlist)

for i in range (6,11):
    current = f1[i].strip()
    crlist = [int(d) for d in current]
    current_board.append(crlist)

def calculate(panel, nextplayer):
    ouragent = 0
    enemy = 0
    evaluate_ouragent=0
    evaluate_enemy=0
    for r in range (num_tiles):
        for j in range(num_tiles):
            if panel[r][j] == state_of_player:
                ouragent = ouragent + 1
                libertypoint = fatch_liberty_move(panel, r, j)
                evaluate_ouragent = evaluate_ouragent+ouragent + libertypoint
            elif panel[r][j] == 3 - state_of_player:
                enemy = enemy + 1
                librtyoponent = fatch_liberty_move(panel, r, j)
                evaluate_enemy = evaluate_enemy+enemy + librtyoponent

    evaluate = evaluate_ouragent - evaluate_enemy
    if nextplayer == state_of_player:
        return evaluate
    return -1 * evaluate

def delete_unused_block(panel, block_state):
    killed_blocks = fatch_useless_blocks(panel, block_state)
    if not killed_blocks:
        return panel
    new_panel = delete_definite_block(panel, killed_blocks)
    return new_panel


def delete_definite_block(panel, locations):
    for block in locations:
        panel[block[0]][block[1]] = 0
    return panel

def fatch_useless_blocks(panel, block_state):
    
    killed_blocks = []
    
    for i in range(num_tiles):
        for j in range(num_tiles):
            if panel[i][j] == block_state:
                if not fatch_liberty_move(panel, i, j):
                    killed_blocks.append((i, j))
    return killed_blocks





def member_ally_frame(panel, i, j):
    position = (i,j)
    array = [(i, j)]
    neighbors_members = []
    while len(array)>0:
        piece = array.pop()
        neighbors_members.append(piece)
        neighbor_allies = get_boardering_node(panel, piece[0], piece[1])
        for neighbor in neighbor_allies:
            if neighbor not in array and neighbor not in neighbors_members:
                array.append(neighbor)
    return neighbors_members

def fatch_adjecent_neighbor(panel, row, column):
    
    neighbors = []
    

    if row > 0:
        neighbors.append((row - 1, column))
    if row < len(panel) - 1:
        neighbors.append((row + 1, column))
    if column > 0:
        neighbors.append((row, column - 1))
    if column < len(panel) - 1:
        neighbors.append((row, column + 1))
    return neighbors

#From the neighbours of a tile we check which of these are allys of our tile by seeing if their colour is the same (i.e 1 for black and 2 for white)
def get_boardering_node(panel, row, column):
    neighbors = fatch_adjecent_neighbor(panel, row, column)
    neighbors_group = []

    for piece in neighbors:

        if panel[piece[0]][piece[1]] == panel[row][column]:
            neighbors_group.append(piece)
    return neighbors_group

#Checks if we can place and how many empty places we can place our tile by chceking the allies.
def fatch_liberty_move(panel, row, column):
    count = 0

    neighbors_members = member_ally_frame(panel, row, column)
    for member in neighbors_members:
        neighbors = fatch_adjecent_neighbor(panel, member[0], member[1])
        for block in neighbors:

            if panel[block[0]][block[1]] == 0:
                count = count + 1
    return count


def fatch(panel, row, column):
    if panel[row][column] == 0:
        return None
    else:
        return 1


def reasonable_move(panel, row, column, ouragent, previous_panel):
    
    panel2 = copy.deepcopy(panel)
    
    panel2[row][column] = ouragent
    
    killed_stone = fatch_useless_blocks(panel2, 3 - ouragent)
    
    panel2 = delete_unused_block(panel2, 3 - ouragent)
    if fatch(panel, row, column) is None and fatch_liberty_move(panel2, row, column) >= 1 and not (
            killed_stone and ko_checker(previous_panel, panel2)):
        return True




def auth_moves(panel, ouragent, previous_panel):
    moves = []
    for row in range(num_tiles):
        for column in range(num_tiles):
            if reasonable_move(panel, row, column, ouragent, previous_panel):
                moves.append((row, column))
    return moves


def ko_checker(previous_panel, panel):
    for r in range(num_tiles):
        for c in range(num_tiles):
            if panel[r][c] != previous_panel[r][c]:
                return False
    return True


def minMaxAlgo(game_position, depth_value, block, prv_panel):
    top_steps = []
    top_performance = None
    alpha = -sys.maxsize
    beta = sys.maxsize
    game_position2 = copy.deepcopy(game_position)
    prv_panel2 = copy.deepcopy(prv_panel)
    upcoming_position = copy.deepcopy(game_position)

    fl = 1
    
    valid_moves = auth_moves(game_position, block, prv_panel2)

    for possible_move in valid_moves :

        fl = fl + 1
        prv_panel2 = copy.deepcopy(upcoming_position)

        upcoming_position[possible_move[0]][possible_move[1]] = block
        upcoming_position = delete_unused_block(upcoming_position, 3 - block)
        evaluate = calculate(upcoming_position, 3 - block)

        evaluation = minMaxAlgo2(upcoming_position, depth_value, alpha, beta, evaluate, 3 - block, prv_panel2)

        upcoming_position = copy.deepcopy(game_position2)
        
        our_best_outcome = -1 * evaluation
        
        if len(top_steps)==0 or our_best_outcome > top_performance:

            top_steps = [possible_move]
            top_performance = our_best_outcome

            alpha = top_performance

        elif our_best_outcome == top_performance:

            top_steps.append(possible_move)
    return top_steps


def minMaxAlgo2(panel, depth_value, alpha, beta, evaluate, nextplayer, prv_panel):
    if depth_value == 0:
        return evaluate
    top_till = evaluate
    game_position2 = copy.deepcopy(panel)
    prv_panel2 = copy.deepcopy(prv_panel)
    upcoming_position = copy.deepcopy(panel)
    
    valid_moves = auth_moves(panel, nextplayer, prv_panel2)

    for possible_move in valid_moves:

        prv_panel2 = copy.deepcopy(upcoming_position)

        upcoming_position[possible_move[0]][possible_move[1]] = nextplayer
        upcoming_position = delete_unused_block(upcoming_position, 3 - nextplayer)

        evaluate = calculate(upcoming_position, 3 - nextplayer)

        evaluation = minMaxAlgo2(upcoming_position, depth_value - 1, alpha, beta, evaluate, 3 - nextplayer, prv_panel2)

        upcoming_position = copy.deepcopy(game_position2)

        our_result = -1 * evaluation
        if our_result > top_till:
            top_till = our_result
        if nextplayer == 3 - state_of_player:
            if top_till > beta:
                beta = top_till

            outcome_for_player = -1 * top_till
            if outcome_for_player < alpha:
                return top_till
        elif nextplayer == state_of_player:
            if top_till > alpha:
                alpha = top_till

            outcome_for_opp = -1 * top_till
            if outcome_for_opp < beta:
                return top_till

    return top_till


def searchPerfectMove(panel, previous_panel):
    movValue = minMaxAlgo(panel, 2, state_of_player, previous_panel)
    return movValue
   


if current_board == [[0]*5]*5 and state_of_player==1:
    a=[(2,2)]

else:
    
    a = searchPerfectMove(current_board, previous_board)
    
    print(a)
    
if a == []:

    f2.write("PASS")
else:
    rand_best = random.choice(a)
    f2.write("%d%s%d" % (rand_best[0], ",", rand_best[1]))

    end = time.time()
print(f'total time of evaluation: {end-start}')
