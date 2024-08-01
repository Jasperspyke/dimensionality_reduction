import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import weiqi as wq
import sys
import time


x = np.zeros([19, 19], dtype=np.int8)
def is_group_in_atari(x, loc, group=None, liberties=None):
    loc = np.asarray(loc, dtype=int)

    # Initialize sets if None
    if group is None:
        group = set()
    if liberties is None:
        liberties = set()

    # Check if location is valid and belongs to the group
    if not (0 <= loc[0] < x.shape[0] and 0 <= loc[1] < x.shape[1]):
        return False
    if x[loc[0], loc[1]] == 0:
        return False
    if tuple(loc) in group:
        return False

    # Add current location to the group
    group.add(tuple(loc))

    # Check adjacent positions
    color = x[loc[0], loc[1]]
    for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        new_loc = loc + dir
        if 0 <= new_loc[0] < x.shape[0] and 0 <= new_loc[1] < x.shape[1]:
            new_color = x[new_loc[0], new_loc[1]]
            if new_color == 0:
                liberties.add(tuple(new_loc))
            elif new_color == color:
                is_group_in_atari(x, new_loc, group, liberties)

    return len(liberties) == 1

def action(x, loc, color, move_count):
    x[loc[0], loc[1]] = color
    last_move = loc[0], loc[1]
    y = process_move(x, last_move, color)
    return y
def process_move(x, loc, color):
    new_board = x.copy()
    for i in range(19):
        for j in range(19):
            if x[i, j] and not has_liberty(x, tuple((i, j))):
                new_board[i, j] = 0
    new_board[loc[0], loc[1]] = color

    return new_board
def would_be_liberty(x, action, color):
    # Check if the action is a legal move, or if it would have no liberties

    if not x[action[0], action[1]] == 0:
        raise SpaceOccupiedError()

    action = np.array(action, dtype=int)
    x[action[0], action[1]] = color
    group = [action]
    visited = set()  # Keep track of visited locations

    while group:
        loc = group.pop()

        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_loc = loc + direction
            if 0 <= new_loc[0] < x.shape[0] and 0 <= new_loc[1] < x.shape[1]:  # Boundary check
                if tuple(new_loc) in visited:
                    continue

                visited.add(tuple(new_loc))

                if x[new_loc[0], new_loc[1]] == 0:
                    x[action[0], action[1]] = 0  # Restore the board before returning
                    return True
                elif x[new_loc[0], new_loc[1]] == x[loc[0], loc[1]]:
                    group.append(new_loc)

    x[action[0], action[1]] = 0  # Restore the board before returning
    return False

def detect_atari_groups(x):
    atari_groups = np.zeros(x.shape, dtype=int)
    for i in range(19):
        for j in range(19):
            if x[i, j] != 0:
                if is_group_in_atari(x, (i, j)):
                    atari_groups[i, j] = 1
    return atari_groups
def has_liberty(board, loc, visited=None):
    # Check if a group has any liberties
    if visited is None:
        visited = set()
    if type(visited) is not set:
        return True


    i, j = loc
    if (i, j) in visited:
        return False

    visited.add((i, j))

    color = board[i, j]
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
            if board[nx, ny] == 0:
                return True
            if board[nx, ny] == color and (nx, ny) not in visited:
                if has_liberty(board, (nx, ny), visited):
                    return True

    return False


def has_liberty(board, loc):
    stack = [loc]
    visited = set()
    i, j = loc
    color = board[i, j]

    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue

        visited.add((x, y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                if board[nx, ny] == 0:
                    return True
                if board[nx, ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

    return False


def legal_move_array(x, color):

    legal_moves = 1 - np.abs(x)  # Initialize all unoccupied moves to legal
    libertiless = []

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):

            if x[i, j] == 0 and not would_be_liberty(x, (i, j), color):
                libertiless.append((i, j))

    # Set all libertiless to False initially
    atari = detect_atari_groups(x)
    for loc in libertiless:
        legal_moves[loc] = check_eye_move(x, loc, color, atari)

    return legal_moves
