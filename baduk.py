import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import weiqi as wq
import sys
import time
from inspect import stack
time_consuming_functions = {
    'process_move': 0,
    'is_group_in_atari': 0,
    'has_liberty': 0,
    'would_be_liberty': 0,
    'check_eye_move': 0,
    'action': 0,
    'detect_atari_groups': 0,
    'legal_move_array': 0,
    'is_stone_in_atari': 0,
    'check_eye_move': 0

}
x = np.zeros([19, 19], dtype=np.int8)
import numpy as np
import time


def is_group_in_atari(x, loc, group=None, liberties=None):
    global time_consuming_functions
    t0 = time.perf_counter()

    # Use a list instead of a set for group and liberties
    if group is None:
        group = []
    if liberties is None:
        liberties = []

    # Stack for depth-first search
    stack = [np.asarray(loc, dtype=int)]
    color = x[loc[0], loc[1]]
    shape = x.shape

    while stack:
        current = stack.pop()

        if tuple(current) in group:
            continue

        group.append(tuple(current))

        # Check adjacent positions
        for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            new_loc = current + dir
            if 0 <= new_loc[0] < shape[0] and 0 <= new_loc[1] < shape[1]:
                new_color = x[new_loc[0], new_loc[1]]
                if new_color == 0:
                    if tuple(new_loc) not in liberties:
                        liberties.append(tuple(new_loc))
                        if len(liberties) > 1:
                            t1 = time.perf_counter()
                            time_consuming_functions['is_group_in_atari'] += t1 - t0
                            return False
                elif new_color == color and tuple(new_loc) not in group:
                    stack.append(new_loc)

    t1 = time.perf_counter()
    time_consuming_functions['is_group_in_atari'] += t1 - t0
    return len(liberties) == 1

def action(x, loc, color, move_count):
    global time_consuming_functions
    t0 = time.perf_counter()
    # Update the board with the new move
    x[loc[0], loc[1]] = color
    last_move = (loc[0], loc[1])

    # Process captures return the updated board
    y = process_move(x, last_move, color)
    t1 = time.perf_counter()
    time_consuming_functions['action'] += t1 - t0
    return y


def identify_groups(board):
    """Identifies groups of stones on the board.

    Args:
        board: A 19x19 numpy array representing the Go board.

    Returns:
        A list of groups, where each group is a list of coordinates.
    """

    global time_consuming_functions
    t0 = time.perf_counter()

    groups = []
    visited = set()

    for row in range(19):
        for col in range(19):
            if board[row, col] != 0 and (row, col) not in visited:
                group = []
                stack = [(row, col)]
                visited.add((row, col))

                while stack:
                    current_pos = stack.pop()
                    group.append(current_pos)

                    for right, left, down, up in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor_pos = (current_pos[0] + down, current_pos[1] + right)
                        if 0 <= neighbor_pos[0] < 19 and 0 <= neighbor_pos[1] < 19:
                            if board[neighbor_pos] == board[current_pos] and neighbor_pos not in visited:
                                stack.append(neighbor_pos)
                                visited.add(neighbor_pos)

                groups.append(group)

    t1 = time.perf_counter()
    time_consuming_functions['identify_groups'] += t1 - t0
    return groups
def process_move(x, loc, color):
    t0 = time.perf_counter()
    # Create a copy of the board to avoid modifying the original
    new_board = x.copy()

    # Remove stones with no liberties
    for i in range(19):
        for j in range(19):
            if x[i, j] and not has_liberty(x, (i, j)):
                new_board[i, j] = 0

    # Set the new move on the board
    new_board[loc[0], loc[1]] = color

    t1 = time.perf_counter()
    time_consuming_functions['process_move'] += t1 - t0
    return new_board
def would_be_liberty(x, action, color):
    global time_consuming_functions
    t0 = time.perf_counter()

    if x[action[0], action[1]] != 0:
        raise SpaceOccupiedError()

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

import numpy as np

import numpy as np

def detect_atari_groups(board):
    global time_consuming_functions
    t0 = time.perf_counter()
    """Detects Atari groups on a Go board efficiently.

    Args:
        board: A 19x19 numpy array representing the Go board.

    Returns:
        A 19x19 numpy array indicating whether each position is in an Atari group.
    """

    atari_groups = np.zeros_like(board, dtype=int)

    for i in range(19):
        for j in range(19):
            if board[i, j] != 0 and is_stone_in_atari(board, (i, j)):
                atari_groups[i, j] = 1
    t1 = time.perf_counter()
    time_consuming_functions['detect_atari_groups'] += t1 - t0
    return atari_groups


def is_stone_in_atari(board, position):
    """Checks if a stone is in atari.

    Args:
        board: A 19x19 numpy array representing the Go board.
        position: A tuple (row, col) representing the stone's position.

    Returns:
        True if the stone is in atari, False otherwise.
    """

    color = board[position]
    liberties = 0
    visited = set()
    queue = [position]

    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if board[nx, ny] == 0:
                    liberties += 1
                elif board[nx, ny] == color and (nx, ny) not in visited:
                    queue.append((nx, ny))

    return liberties == 1
def check_eye_move(board, action, color, atari_groups):
    """Checks if a move is an eye move.

    An eye move is a move that creates a true eye for the player.

    Args:
        board: A 19x19 numpy array representing the Go board.
        action: A tuple (row, col) representing the move to check.
        color: The color of the move (1 or -1).
        atari_groups: A 19x19 numpy array indicating Atari groups.

    Returns:
        True if the move is an eye move, False otherwise.
    """

    global time_consuming_functions
    t0 = time.perf_counter()

    # Check if any adjacent position is in atari
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_loc = np.array(action) + np.array((dr, dc))
        if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19 and atari_groups[new_loc[0], new_loc[1]]:
            return False

    # Check if all surrounding positions are either the same color or empty
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue  # Skip the center position
            new_loc = np.array(action) + np.array((dr, dc))
            if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19 and board[new_loc[0], new_loc[1]] != color:
                return False

    time_consuming_functions['check_eye_move'] += time.perf_counter() - t0
    return True

def has_liberty(board, location):
    global time_consuming_functions
    t0 = time.perf_counter()
    """Checks if a group has any liberties using an iterative approach.

    Args:
        board: A 19x19 numpy array representing the Go board.
        location: A tuple (row, col) representing the position to check.

    Returns:
        True if the group has at least one liberty, False otherwise.
    """

    stack = [location]
    visited = set()
    row, col = location
    color = board[row, col]

    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue

        visited.add((x, y))

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = x + dr, y + dc
            if 0 <= nr < 19 and 0 <= nc < 19:
                if board[nr, nc] == 0:
                    return True
                if board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
    time_consuming_functions['has_liberty'] += time.perf_counter() - t0

    return False

def legal_move_array(board, color):

    global time_consuming_functions
    t0 = time.perf_counter()
    """Calculates legal moves for a given player.

    Args:
        board: A 19x19 numpy array representing the Go board.
        color: The color of the player (1 or -1).

    Returns:
        A 19x19 numpy array indicating legal moves.
    """

    legal_moves = 1 - np.abs(board)  # Initialize all unoccupied moves to legal
    libertiless = []


    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 0:
                if not would_be_liberty(board, (i, j), color):
                    libertiless.append((i, j))

    # Set all libertiless to False initially

    # THE PROBLEM IS HERE
    # this is the slow
    atari_groups = detect_atari_groups(board)
    for loc in libertiless:
        legal_moves[loc] = check_eye_move(board, loc, color, atari_groups)
    t1 = time.perf_counter()
    time_consuming_functions['legal_move_array'] += t1 - t0

    return legal_moves
