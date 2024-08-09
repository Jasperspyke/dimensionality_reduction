import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
import inspect
from inspect import stack, getargvalues
time_consuming_functions = {
    'process': 0,
    'would_be_liberty': 0,
    'check_eye_move': 0,
    'action': 0,
    'count_liberties': 0,
    'groupify': 0,
    'friend': 0


}
ko_states = set()


def encode(x):
    board = x.copy()
    board += 1
    hashed = ''
    for i in range(19):
        for j in range(19):
            hashed += (str(board[i, j]))
    return hashed

def fancy_string(matrix):


    num_rows, num_cols = matrix.shape

    # Convert the matrix to string type
    matrix_str = matrix.astype(str)

    # Create column index (as a row) and add as the first row
    col_index = np.arange(num_cols).astype(str)
    header_row = np.insert(col_index, 0, '')
    matrix_with_col_index = np.vstack([header_row, np.column_stack([np.arange(num_rows).astype(str), matrix_str])])

    # Create DataFrame
    df = pd.DataFrame(matrix_with_col_index)

    return df.to_string(index=False, header=False) + '\n' + '___________________________________________________________'

class IllegalMoveError(Exception):
    def __init__(self, message="Illegal move!"):
        super().__init__(message)
class KoError(Exception):
    def __init__(self, message="Ko violation detected!"):
        super().__init__(message)
class SpaceOccupiedError(Exception):
    def __init__(self, message="Space Already Taken!"):
        super().__init__(message)

def action(x, loc, color, tracker):
    global time_consuming_functions
    t0 = time.perf_counter()
    # Update the board with the new move
    last_move = (loc[0], loc[1])
    legals = tracker.white.legal_moves if color == -1 else tracker.black.legal_moves
    if not legals[loc[0], loc[1]]:
        fancy_print(x)
        fancy_print(legals)
        raise IllegalMoveError
    x[loc[0], loc[1]] = color


    # Process captures return the updated board
    return x, tracker



def would_be_liberty(x, action, color):
    global time_consuming_functions
    t0 = time.perf_counter()

    if x[action[0], action[1]] != 0:
        # should maybe error here
        return False

    x[action[0], action[1]] = color
    stack = [action]
    visited = set()

    while stack:
        row, col = stack.pop()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < x.shape[0] and 0 <= new_col < x.shape[1]:
                if x[new_row, new_col] == 0:
                    x[action[0], action[1]] = 0
                    time_consuming_functions['would_be_liberty'] += time.perf_counter() - t0
                    return True
                if x[new_row, new_col] == color and (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    stack.append((new_row, new_col))

    x[action[0], action[1]] = 0

    time_consuming_functions['would_be_liberty'] += time.perf_counter() - t0
    return False


def fancy_print(matrix):
    matrix = np.fliplr(np.flipud(np.fliplr(matrix.transpose())))
    num_rows, num_cols = matrix.shape

    # Convert the matrix to string type
    matrix_str = matrix.astype(str)

    # Create column index (as a row) and add as the first row
    col_index = np.arange(num_cols).astype(str)
    header_row = np.insert(col_index, 0, '')
    matrix_with_col_index = np.vstack([header_row, np.column_stack([np.arange(num_rows).astype(str), matrix_str])])

    # Create DataFrame
    df = pd.DataFrame(matrix_with_col_index)

    # Print DataFrame
    print(df.to_string(index=False, header=False))
    print('___________________________________________________________')

def check_eye_move(x, action, color, tracker):
    global time_consuming_functions
    t0 = time.perf_counter()
    enemy = tracker.white if color == 1 else tracker.black
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_loc = np.array(action) + np.array(direction)
        if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19:
            if x[new_loc[0], new_loc[1]] == -color:
                for group in enemy.groups:
                    if tuple(new_loc) in group:
                        if count_liberties(group, x) == 1:
                            time_consuming_functions['check_eye_move'] += time.perf_counter() - t0
                            return True

    time_consuming_functions['check_eye_move'] += time.perf_counter() - t0
    return False

# group is a set of tuples
def count_liberties(group, board):
    t0 = time.perf_counter()
    liberties = set()
    for stone in group:
        x, y = stone

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 19 and 0 <= new_y < 19 and board[new_x, new_y] == 0:
                liberties.add((new_x, new_y))
    t1 = time.perf_counter()
    time_consuming_functions['count_liberties'] += t1 - t0
    return len(liberties)

class ColorCategory:
    def __init__(self, color):
        self.color = color
        self.legal_moves = np.ones([19, 19], dtype=int)
        self.groups = []
        self.liberties = []

    def __repr__(self):
        return self.color + ' group: ' + str(self.groups)

    def __contains__(self, item):
        return item in self.group

    def update_legal_moves(self, board):



        self.legal_moves = new_legals

class BoardTracker:
    def __init__(self):
        self.white = ColorCategory('white')
        self.black = ColorCategory('black')
        self.legal_moves = np.ones([19, 19], dtype=int)



    def __refsdfpr__(self):

        arr = np.zeros([19, 19], dtype=int)
        s1 = 'White: ' + str(self.white) + 'Black: ' + str(self.black)
        for i, group in enumerate(self.black.groups):
            for loc in group:
                arr[loc] += i + 1
        for i, group in enumerate(self.white.groups):
            for loc in group:
                arr[loc] -= i + 1



        return fancy_string(arr)

def groupify(tracker, loc, color):
    t0 = time.perf_counter()
    connections = 0
    new_group = {loc}
    structure = tracker.white.groups if color == -1 else tracker.black.groups

    for dir in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        dir = np.array(dir)
        new_loc = tuple(dir + loc)

        if any(new_loc in s for s in structure):

            for s in structure:
                if new_loc in s:
                    new_group = new_group.union(s)
                    structure.remove(s)
                    connections += 1
    structure.append(new_group)
    t1 = time.perf_counter()
    time_consuming_functions['groupify'] += t1 - t0
    return tracker, connections


def process(board, loc, color, tracker, ko_states):
    global time_consuming_functions
    t0 = time.perf_counter()
    tracker, connections = groupify(tracker, loc, color)
    friend = tracker.white if color == -1 else tracker.black
    enemy = tracker.white if color == 1 else tracker.black
    friend.liberties = [0] * len(friend.groups)
    new_board = board.copy()
    self_liberties = 0
    adjacent_allies = []
    adjacent_enemies = []

    for dir in ((1, 0), (-1, 0), (0, -1), (0, 1)):
        new_loc = np.array(loc) + np.array(dir)
        if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19:
            if board[new_loc[0], new_loc[1]] == -color:
                adjacent_enemies.append(new_loc)
            elif board[new_loc[0], new_loc[1]] == color:
                adjacent_allies.append(new_loc)
            else:
                self_liberties += 1


    for i, stone in enumerate(adjacent_enemies):
        stone = tuple(stone)
        for group in enemy.groups:
            if stone in group:
                lib = count_liberties(group, new_board)
                if lib == 0:
                    for location in group:
                        new_board[location] = 0
                    enemy.groups.remove(group)
                    for st in group:
                        enemy.legal_moves[st] = 1
                        friend.legal_moves[st] = 1
                else:
                    enemy.liberties[i] = lib

    for i, stone in enumerate(adjacent_allies):
        stone = tuple(stone)
        if not any(stone in group for group in friend.groups):
            sys.exit(1)
        for group in friend.groups:
            if stone in group:
                #lib = count_liberties(group, new_board)
                try:
                    lib = friend.liberties[i] - connections
                except IndexError:
                    pass

                if lib == 0:
                    raise SelfCaptureError
                else:
                    try:
                        friend.liberties[i] = lib
                    except IndexError:
                        pass

    encoded = encode(new_board)
    if encoded in ko_states:
        print('Ko violation detected! at ', loc)
        board[loc[0], loc[1]] = 999
        fancy_print(board)
        for j, i in enumerate(ko_states):
            if i == encoded:
                print('Ko state: ')
                print('j is: ', j)
                ko_states = list(ko_states)
                print(ko_states[j])
                print('encoded is: ', encoded)
                print(len(ko_states))
        raise KoError
    ko_states.add(encoded)

    libertiless_spaces = []

    # look around the new stone for libertiless spaces
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_loc = np.array(loc) + np.array(direction)
        if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19:
            if new_board[new_loc[0], new_loc[1]] == 0:
                if not would_be_liberty(new_board, new_loc, color):
                    libertiless_spaces.append(tuple(new_loc))
    friend_start = time.perf_counter()
    friend.legal_moves[loc] = 0
    for loc in libertiless_spaces:
        friend.legal_moves[loc] = 0
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_loc = np.array(loc) + np.array(direction)
            if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19:
                if new_board[new_loc[0], new_loc[1]] == -color:
                    for group in enemy.groups:
                        if tuple(new_loc) in group:
                            if count_liberties(group, new_board) == 1:
                                friend.legal_moves[loc] = 1
                                break

    enemy.legal_moves[loc] = 0
    for loc in libertiless_spaces:
        enemy.legal_moves[loc] = 0
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_loc = np.array(loc) + np.array(direction)
            if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19:
                if new_board[new_loc[0], new_loc[1]] == color:
                    for group in friend.groups:
                        if tuple(new_loc) in group:
                            if count_liberties(group, new_board) == 1:
                                enemy.legal_moves[loc] = 1
                                break
    friend_end = time.perf_counter()
    time_consuming_functions['friend'] += friend_end - friend_start
    t1 = time.perf_counter()
    time_consuming_functions['process'] += t1 - t0


    if len(ko_states) > 2:

        ko_states = set(list(ko_states)[-2:])
    return new_board, tracker, ko_states