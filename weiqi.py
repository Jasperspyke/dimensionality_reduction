import numpy as np
import sys

x = np.zeros([19, 19], dtype=int)

import pandas as pd



def fancy_print(matrix):
    df = pd.DataFrame(matrix)
    print(df.to_string(index=False, header=False))
    print('___________________________________________________________')



def process_move(x, last_move):
    has_liberties = np.full(fill_value=False, shape=[19, 19], dtype=bool)
    empty_spaces = np.where(x==0)
    has_liberties[empty_spaces] = True
    empty_spaces = list(zip(empty_spaces[0], empty_spaces[1]))
    for i in range(0, 19):
        for j in range(0, 19):
            here = x[i, j]
            if (i, j) in empty_spaces:
                if i != 0:
                    has_liberties[i-1, j] = True
                if i != 18:
                    has_liberties[i+1, j] = True
                if j != 0:
                    has_liberties[i, j-1] = True
                if j != 18:
                    has_liberties[i, j+1] = True

            if has_liberties[i, j] == True:
                # stones with liberties grant liberties to same-colored neighbors
                if i != 0:
                    if x[i-1, j] == here:
                        has_liberties[i-1, j] = True
                if i != 18:
                    if x[i+1, j] == here:
                        has_liberties[i+1, j] = True
                if j != 0:
                    if x[i, j-1] == here:
                        has_liberties[i, j-1] = True
                if j != 18:
                    if x[i, j+1] == here:
                        has_liberties[i, j+1] = True

    y = np.where(has_liberties, x, 0)
    new_board = np.where(y != 0, x, 0)
    new_board[last_move[0], last_move[1]] = last_move[2]
    return new_board

def plot_board(x, color=1, territory=None, ring_moves=None, move_count=None, stone_opacity=1.0, fig=None, ax=None):

    if color == 1:
        color = 'black'
    else:
        color = 'white'
    if fig is None:

        fig, ax = plt.subplots(figsize=[12, 8])
    fig.patch.set_facecolor((1, 1, .8))
    ax.set_xlim([0, 24])
    ax.set_ylim([0, 18])
    plt.xticks(ticks=np.arange(1,20) * 24/19 -0.633, labels=np.arange(1, 20))
    plt.yticks(ticks=np.arange(1,20) * 18.45/20 - 0.2, labels=np.arange(1, 20))

    if x.any():

        black_state = list(np.argwhere(x == 1))
        white_state = list(np.argwhere(x == -1))
        for idx, item in enumerate(black_state):
            item = tuple((item[0], item[1], 1))
            black_state[idx] = item
        for idx, item in enumerate(white_state):
            item = tuple((item[0], item[1], -1))
            white_state[idx] = item
        show_stones(fig, ax, black_state, opacity=stone_opacity)
        show_stones(fig, ax, white_state, opacity=stone_opacity)
        if ring_moves is not None:
            last_move = ring_moves[:move_count + 1][-1]
            show_ring(fig, ax, last_move, color, ring_radius=2, ring_thickness=1.125, alpha=1)
        if territory is not None:
            display_all_territory(fig, ax, territory)
    else:
        print('problem! X is:', x)

    return fig, ax


def get_liberties(x):
    has_liberties = np.full(fill_value=False, shape=[19, 19], dtype=bool)
    empty_spaces = np.where(x==0)
    has_liberties[empty_spaces] = True
    empty_spaces = list(zip(empty_spaces[0], empty_spaces[1]))
    for i in range(0, 19):
        for j in range(0, 19):
            here = x[i, j]
            if (i, j) in empty_spaces:
                if i != 0:
                    has_liberties[i-1, j] = True
                if i != 18:
                    has_liberties[i+1, j] = True
                if j != 0:
                    has_liberties[i, j-1] = True
                if j != 18:
                    has_liberties[i, j+1] = True


            if has_liberties[i, j] == True:
                # stones with liberties grant liberties to same-colored neighbors
                if i != 0:
                    if x[i-1, j] == here:
                        has_liberties[i-1, j] = True
                if i != 18:
                    if x[i+1, j] == here:
                        has_liberties[i+1, j] = True
                if j != 0:
                    if x[i, j-1] == here:
                        has_liberties[i, j-1] = True

                if j != 18:
                    if x[i, j+1] == here:

                        has_liberties[i, j+1] = True
    return has_liberties

def SpaceOccupiedError(Exception):
    def __init__(self, message="Space occupied!"):
        super().__init__(message)

def is_liberty(x, action, color):
    if not x[action[0], action[1]] == 0:
        raise SpaceOccupiedError()
    action = np.array(action, dtype=int)
    x[action[0], action[1]] = color
    group = [action]

    while group:
        loc = group.pop()
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_loc = loc + direction
            if 0 <= new_loc[0] < x.shape[0] and 0 <= new_loc[1] < x.shape[1]:  # Boundary check
                if x[new_loc[0], new_loc[1]] == 0:
                    x[action[0], action[1]] = 0  # Restore the board before returning
                    return True
                elif x[new_loc[0], new_loc[1]] == x[loc[0], loc[1]]:
                    group.append(new_loc)

    x[action[0], action[1]] = 0  # Restore the board before returning
    return False


def is_liberty_recursive(board, loc):

    i, j = loc[0], loc[1]
    visited = set()
    if tuple((i, j)) in visited:
        return False
    visited.add((i, j))

    color = board[i, j]
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
            if board[nx, ny] == 0:
                return True
            if board[nx, ny] == color and is_liberty_recursive(board, nx, ny, visited):
                return True

    return False


def attempt_eye_capture(x, action, color):
    if not x[action[0], action[1]] == 0:
        print('Space is already taken at ' + str(action) + ' by ' + str(x[action[0], action[1]]))
        raise SpaceOccupiedError
    x[action] = color
    has_liberties = np.full(fill_value=False, shape=[19, 19], dtype=bool)
    empty_spaces = np.where(x == 0)
    has_liberties[empty_spaces] = True
    empty_spaces = list(zip(empty_spaces[0], empty_spaces[1]))

    for i in range(0, 19):
        for j in range(0, 19):
            here = x[i, j]
            if (i, j) in empty_spaces:
                if i != 0:
                    has_liberties[i - 1, j] = True
                if i != 18:
                    has_liberties[i + 1, j] = True
                if j != 0:
                    has_liberties[i, j - 1] = True
                if j != 18:
                    has_liberties[i, j + 1] = True

    assert has_liberties[action] == False

    for i in range(0, 19):
        for j in range(0, 19):
            if has_liberties[i, j] == True:
                # stones with liberties grant liberties to same-colored neighbors
                if i != 0:
                    if x[i - 1, j] == here:
                        has_liberties[i - 1, j] = True
                if i != 18:
                    if x[i + 1, j] == here:
                        has_liberties[i + 1, j] = True
                if j != 0:
                    if x[i, j - 1] == here:
                        has_liberties[i, j - 1] = True
                if j != 18:
                    if x[i, j + 1] == here:
                        has_liberties[i, j + 1] = True

    # update the board by removing captured stones
    y = np.where(has_liberties, x, 0)
    new_board = np.where(y != 0, x, 0)
    # re-add the action and check if it has liberties
    new_board[action] = color
    lib = get_liberties(new_board)
    if lib[action] == False:
        print('Illegal move!')
        raise ValueError

    return new_board


def attempt_eye_capture_white(x, action):
    assert x[action] == 0
    x[action] = -1
    has_liberties = np.full(fill_value=False, shape=[19, 19], dtype=bool)
    empty_spaces = np.where(x==0)
    has_liberties[empty_spaces] = True
    empty_spaces = list(zip(empty_spaces[0], empty_spaces[1]))
    for i in range(0, 19):
        for j in range(0, 19):
            here = x[i, j]
            if (i, j) in empty_spaces:
                if i != 0:
                    has_liberties[i-1, j] = True
                if i != 18:
                    has_liberties[i+1, j] = True
                if j != 0:
                    has_liberties[i, j-1] = True
                if j != 18:
                    has_liberties[i, j+1] = True

    assert has_liberties[action] == False
    for i in range(0, 19):
        for j in range(0, 19):
            if has_liberties[i, j] == True:
                # stones with liberties grant liberties to same-colored neighbors
                if i != 0:
                    if x[i-1, j] == here:
                        has_liberties[i-1, j] = True
                if i != 18:
                    if x[i+1, j] == here:
                        has_liberties[i+1, j] = True
                if j != 0:
                    if x[i, j-1] == here:
                        has_liberties[i, j-1] = True
                if j != 18:
                    if x[i, j+1] == here:
                        has_liberties[i, j+1] = True
    # update the board by removing captured stones
    p.where(has_liberties, x, 0)
    new_board = np.where(y != 0, x, 0)
    # re-add the action and check if it has liberties
    new_board[action] = -1
    print('move!')
    lib = get_liberties(new_board)
    if lib[action] == False:
        fancy_print(lib)
        print('Illegal move!')
        raise ValueError

import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sys

# load png images in matplotlib and display them
go_board = plt.imread(r'C:\Users\Jasper\Documents\pythonProject31\static\go_board.png')
black_piece = plt.imread(r'C:\Users\Jasper\Documents\pythonProject31\static\go_black.png')
white_piece = plt.imread(r'C:\Users\Jasper\Documents\pythonProject31\static\go_white.png')
black_territory = plt.imread(r'C:\Users\Jasper\Documents\pythonProject31\static\obama_2.png')
white_territory = plt.imread(r'C:\Users\Jasper\Documents\pythonProject31\static\biden_2.png')

def move(loc, x, color, state, ko_states):
    captures = None
    if not x[loc[0], loc[1]] == 0:
        raise ValueError('Space is already taken')
    if not is_liberty_recursive(x, loc, 'white' if color == -1 else 'black'):
        x = attempt_eye_capture(x, loc, color)
    x[loc[0], loc[1]] = color
    y = process_move(x)
    return y


def preprocess(img, biden=False):
    # Crop the image to be square
    height, width = img.shape[:2]
    assert width >= height
    half = (width - height) // 2
    cropped_img = img[:, half:half + height]

    # Convert to grayscale
    gray_img = np.dot(cropped_img[..., :3], [0.2989, 0.5870, 0.1140])

    background = gray_img > 0.1
    if biden:
        gray_img *= 1.2
        gray_img = np.clip(gray_img, 0, 1)
        background = gray_img > 0.6
        background = ~background
    gray_img = np.dstack([gray_img, gray_img, gray_img, background])

    # Display the grayscale image
    return gray_img


black_territory = preprocess(black_territory)
white_territory = preprocess(white_territory, biden=True)
white_territory = np.clip(white_territory, 0, 0.75)


def fancy_print(matrix):
    df = pd.DataFrame(matrix)
    print(df.to_string(index=False, header=False))
    print('___________________________________________________________')

# minuses 1 from each element in a tuple or list of tuples
def minus_one(loc):
    if isinstance(loc, tuple):
        return tuple([x - 1 for x in loc])
    return [tuple([x - 1 for x in loc]) for loc in loc]
def plus_one(loc):
    if isinstance(loc, tuple):
        return tuple([x + 1 for x in loc])
    return [tuple([x + 1 for x in loc]) for loc in loc]
# crop x dimensions to within 218 and 741
go_board = go_board[16:, 218:741, :]


def draw_ring(radius, thickness, opacity=1.0):
    size = 2 * (radius + thickness) + 1
    ring = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    alpha = int(255 * opacity)

    for r in range(radius, radius + thickness):
        for theta in np.linspace(0, 2 * np.pi, 1000):
            x = int(center[0] + r * np.cos(theta))
            y = int(center[1] + r * np.sin(theta))
            if 0 <= x < size and 0 <= y < size:
                ring[x, y, 0] = 255  # Red channel
                ring[x, y, 1] = 255  # Green channel
                ring[x, y, 2] = 255  # Blue channel
                ring[x, y, 3] = alpha  # Alpha channel

    return ring


def show_ring(fig, ax, placement, color, ring_radius=1, ring_thickness=10, opacity=1.0):
    x, y = placement
    assert 0 <= x < 19 and 0 <= y < 19
    x_intercept = 7
    y_intercept = 6.17
    x_slope = 12.531
    y_slope = 9.31

    x_center = x_intercept + x_slope * x
    y_center = y_intercept + y_slope * y

    ring = draw_ring(int(ring_radius * 100), int(ring_thickness * 100), opacity)
    ring_size = ring.shape[0]
    extent = [
        x_center - ring_size / 200,
        x_center + ring_size / 200,
        y_center - ring_size / 200,
        y_center + ring_size / 200
    ]
    if color == -1:
        for dim in range(0, 3):
            ring[:, :, dim] = 1 - ring[:, :, dim]
    extent = (np.array(extent) * 0.1)
    ax.imshow(ring, extent=extent)


def show_stones(fig, ax, placements, ring=False, count_n_stones=None, opacity=1.0):
    count = 0

    for placement in placements:
        count += 1
        x, y, color = placement
        assert 0 <= x < 19 and 0 <= y < 19

        stone_aspect_ratio = white_piece.shape[1] / white_piece.shape[0]

        x_intercept, y_intercept = 0.75, 0.59
        x_slope, y_slope = 1.25, 0.928

        stone_half_size = 4

        stone = white_piece if color == -1 else black_piece
        x, y = x_intercept + x_slope * x, y_intercept + y_slope * y
        if color != 1:
            y *= 1.01

        ax.imshow(stone, extent=[x - stone_aspect_ratio * stone_half_size,
                                 x + stone_aspect_ratio * stone_half_size,
                                 y - stone_half_size,
                                 y + stone_half_size], alpha=opacity)

    if ring:  # Draw ring around the last stone
        show_ring(fig, ax, [placements[-1][0], placements[-1][1]], 1, ring_radius=2, ring_thickness=1.125,
                  opacity=1)  # Adjust opacity here

    if count_n_stones is not None:
        #def plot_number(fig, ax, placement, number, color='white'):
        for j, i in enumerate(range(-count_n_stones, 0)):
            plot_number(fig, ax, [placements[i][0], placements[i][1]], color=placements[i][2], number=j + 1)


def is_chain_alive(x, loc):


    here = x[loc[0], loc[1]]
    group = [loc]
    eyes = set()
    territory_map = map_territory(x)
    visited = set()  # Add a set to keep track of all visited positions

    if here == 0:
        return False

    while group:
        loc = group.pop()
        if tuple(loc) in visited:  # Skip if already visited
            continue
        visited.add(tuple(loc))  # Mark as visited

        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            direction = np.array(direction)
            loc = np.array(loc)
            new_loc = loc + direction
            if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19:
                # if the group has a liberty, check that it would be territory
                if x[new_loc[0], new_loc[1]] == 0 and territory_map[new_loc[0], new_loc[1]] == here:
                    eyes.add(tuple(new_loc))
                elif x[new_loc[0], new_loc[1]] == here and tuple(new_loc) not in visited:
                    group.append(tuple(new_loc))

    if len(eyes) > 1:
        return True
    else:
        return False



# Assuming get_territory function is defined elsewhere
def is_territory(x, loc):
    if x[loc] != 0:
        return False
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_loc = loc + direction
        if 0 <= new_loc[0] < 19 and 0 <= new_loc[1] < 19:
            if x[new_loc[0], new_loc[1]] != 0:
                return False
    return True

def map_territory(x):
    territory_map = np.zeros([19, 19], dtype=int)
    for i in range(19):
        for j in range(19):
            if is_territory(x, (i, j)):
                territory_map[i, j] = 1
    return territory_map

def show_territory(fig, ax, placement, color):
    x, y = placement
    assert 0 <= x < 19 and 0 <= y < 19
    territory = white_territory if color == -1 else black_territory

    x_intercept = -0.20
    y_intercept = 0.20
    x_slope = 1 * 1.123 * 20 / 18
    y_slope = 1 * 0.84 * 20 / 18



    x1, x2, y1, y2 = x_intercept + x_slope * x, x_intercept + x_slope * (
                x + 1.4), y_intercept + y_slope * y, y_intercept + y_slope * (y + 1)


    ax.imshow(territory, extent=[x1, x2, y1, y2], alpha=True)  # Use alpha=True to respect alpha channel

def get_territory(x):
    territory_map = np.zeros([19, 19], dtype=int)
    score = 0
    for i in range(19):
        for j in range(19):
            if x[i, j] == 0:
                v1 = np.flip(x[:i, j])      # Up direction (column slice above the current position)
                v2 = x[i+1:, j]             # Down direction (column slice below the current position)
                h1 = np.flip(x[i, :j])      # Left direction (row slice left of the current position)
                h2 = x[i, j+1:]             # Right direction (row slice right of the current position)

                territory = None
                for dir in [v1, v2, h1, h2]:
                        adj = np.argwhere(dir != 0)
                        if len(adj) > 0 and x[i, j] == 0:
                            adj = adj[0]
                            if territory is None:
                                territory = dir[adj[0]]
                            elif dir[adj[0]] != territory:
                                    territory = 0
                            else:
                                pass

                if territory:
                    score += territory
                    territory_map[i, j] = territory

    return score, territory_map


class KoError(Exception):
    def __init__(self, message="Ko violation detected!"):
        super().__init__(message)

class IllegalMoveError(Exception):
    def __init__(self, message="Illegal move detected!"):
        super().__init__(message)

def score_game(x, black_prisoners, white_prisoners,komi=6.5):

    score, territory_arr = get_territory(x)
    score += black_prisoners
    score -= white_prisoners
    score -= komi
    winner = np.sign(score)
    return score, territory_arr, winner

def display_all_territory(fig, ax, territory_map):
    for i in range(19):
        for j in range(19):
            if territory_map[i, j] != 0:
                color = territory_map[i, j]
                show_territory(fig, ax, (i, j), color)

def plot_number(fig, ax, placement, number, color='white'):

    color *= -1
    x, y = placement
    assert 0 <= x < 19 and 0 <= y < 19

    x_intercept = 0.71
    y_intercept = 0.58
    x_slope = 1.255
    y_slope = 0.929

    x_pos = x_intercept + x_slope * x
    y_pos = y_intercept + y_slope * y

    ax.text(x_pos, y_pos, str(number), color=color, fontsize=16, ha='center', va='center')



def remove_dead_stones(x):
    new_x = x.copy()
    dead_stones = np.zeros(x.shape, dtype=int)
    for i in range(19):
        for j in range(19):
            if x[i, j] != 0:
                if not is_chain_alive(x, (i, j)):
                    new_x[i, j] = 0
                    dead_stones[(i, j)] = x[i, j]
    return new_x, dead_stones

def end_game(state, x, move_count):
    print('Game over on move ' + str(move_count) + '!')
    y, dead = remove_dead_stones(x)
    fancy_print(dead)
    num_removed_stones = len(np.nonzero(dead)[0])
    print('Removed ' + str(num_removed_stones) + ' dead stones!')
    x = y
    score, territory, winner = score_game(x, 0, 0)
    print('Black score: ' + str(score) + ' points')
    print('Winner: ' + ('Black' if winner == 1 else 'White'))
    print('Scoring...')


    fig, ax = plot_board(x, territory=territory, fig=None, ax=None)
    fig, ax = plot_board(dead, stone_opacity=0.5, fig=fig, ax=ax)
    ax.imshow(go_board, extent=[0, 24, 0, 18], zorder=-1)
    plt.show()
def run_from_list():
    state = [
        (4, 3), (4, 2), (5, 3), (5, 2), (6, 3), (6, 2), (6, 4), (7, 4),
        (6, 5), (7, 5), (6, 6), (7, 6), (6, 7), (7, 7), (5, 5), (6, 8),
        (5, 7), (5, 8), (4, 7), (4, 8), (4, 4), (7, 3), (4, 5), (3, 5),
        (4, 6), (3, 6), (19, 19), (3, 3), (19, 18), (3, 4), (18, 19),
        (3, 7), (5,10), (16, 1), (17, 1), (16, 2), (17, 2), (17, 3),
        (18, 2), (17, 4), (18, 3), (1, 17), (17, 5), (18, 4), (19, 5),
        (19, 4), (19, 3), (18, 5), (19, 1), (1,1), (2, 1), (8,8), (1, 2), (9, 1),
    ]
    x = np.zeros([19, 19], dtype=int)
    ko_states = []
    move_count = 0
    last_player_passed = False

    for action in state:
        action = minus_one(action)
        color = 1 if move_count % 2 == 0 else -1

        if isinstance(action, str) and action.lower() == 'pass':
            if last_player_passed:
                print('Game over!')
                break
            last_player_passed = True
            move_count += 1
            continue
        last_player_passed = False

        y = move(action, x, color, state, ko_states)

        move_count += 1
        n = len(state) - 1
        if move_count == n:
            print('done!')
            end_game(state, y, move_count)

def legal_move_array(x, color):
    legal_moves = 1-np.abs(x)
    libertiless = []
    for i in range(19):
        for j in range(19):
             if x[i, j] == 0 and not is_liberty(x, (i, j), 1):
                libertiless.append((i, j))

    for loc in libertiless:
        try:
            _ = attempt_eye_capture(x, loc, color)
            x[loc[0], loc[1]] = 0
            legal_moves[loc] = True
        except ValueError:
            legal_moves[loc] = False

    return legal_moves

def action(loc, x, color, ko_states):
    if not x[loc[0], loc[1]] == 0:
        raise SpaceOccupiedError('OOPS!')

    legal = legal_move_array(x, color)

    if not legal[loc]:
        raise IllegalMoveError

    x[loc[0], loc[1]] = color
    last_move = loc[0], loc[1], color
    y = process_move(x, last_move=last_move)
    ko_states.add(tuple(map(tuple, y)))
    if tuple(map(tuple, y)) in ko_states:
        raise KoError
    return y, ko_states


def run_from_goban():
    x = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, -1, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    ko_states = set(tuple(map(tuple, x)))
    x, ko_states = action((11, 12), x, 1, ko_states=ko_states)
    x, ko_states = action((11, 13), x, -1, ko_states=ko_states)

    fig, ax = plot_board(x)
    ax.imshow(go_board, extent=[0, 24, 0, 18], zorder=-1)
    plt.show()

run_from_goban()
