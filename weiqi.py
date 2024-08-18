import re
import matplotlib.pyplot as plt
from baduk import *
import os
import sys

# load png images in matplotlib and display them
static = 'static'
go_board = plt.imread(os.path.join(static, 'go_board.png'))[16:, 218:741, :]
black_piece = plt.imread(os.path.join(static, 'go_black.png'))
white_piece = plt.imread(os.path.join(static, 'go_white.png'))
black_territory = plt.imread(os.path.join(static, 'obama_2.png'))
white_territory = plt.imread(os.path.join(static, 'biden_2.png'))



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
        for j, i in enumerate(range(-count_n_stones, 0)):
            plot_number(fig, ax, [placements[i][0], placements[i][1]], color=placements[i][2], number=j + 1)

def show_ring(fig, ax, placement, color, ring_radius=1, ring_thickness=10, opacity=1.0):
    x, y = placement
    y = 18 - y
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
    plt.xticks(ticks=np.arange(1, 20) * 24/19 -0.633, labels=np.arange(1, 20))
    plt.yticks(ticks=np.arange(1, 20) * 18.45/20 - 0.2, labels=np.arange(1, 20))
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
            last_move = ring_moves[:move_count][-1][-1]
            show_ring(fig, ax, last_move, color, ring_radius=2, ring_thickness=1.125, opacity=1)
        if territory is not None:
            display_all_territory(fig, ax, territory)
    else:
        print('problem! X is:', x)
    return fig, ax

class SelfCaptureError(Exception):
    def __init__(self, message="Self Capture Detected!"):
        super().__init__(message)

class KoError(Exception):
    def __init__(self, message="Ko violation detected!"):
        super().__init__(message)

class IllegalMoveError(Exception):
    def __init__(self, message="Illegal move detected!"):
        super().__init__(message)

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

# minuses 1 from each element in a tuple or list of tuples
def minus_one(loc):
    if isinstance(loc, tuple):
        return tuple([x - 1 for x in loc])
    return [tuple([x - 1 for x in loc]) for loc in loc]



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
def is_eye_in_living_shape(x, location):
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
     loc = np.array(location) + np.array(direction)
     if 0 <= loc[0] < x.shape[0] and 0 <= loc[1] < x.shape[1]:
        here = x[loc[0], loc[1]]
        group = [loc]
        eyes = set()
        territory_map = get_territory(x)[1]
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
    return False

def is_chain_alive(x, loc):


    here = x[loc[0], loc[1]]
    group = [loc]
    eyes = set()
    territory_map = get_territory(x)[1]
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
    elif len(eyes) == 1 and is_eye_in_living_shape(x, eyes.pop()):
        return True
    else:
        return False

def is_chain_dead(x, loc):
    if x[loc] == 0:
        return False  # Empty space is not a chain

    color = x[loc]
    directions = ['up', 'down', 'left', 'right', 'diag1', 'diag2']

    for direction in directions:
        if direction == 'up':
            path = x[loc[0]:, loc[1]]
        elif direction == 'down':
            path = np.flip(x[:loc[0] + 1, loc[1]])
        elif direction == 'left':
            path = np.flip(x[loc[0], :loc[1] + 1])
        elif direction == 'right':
            path = x[loc[0], loc[1]:]
        elif direction == 'diag1':
            path = np.diagonal(x, offset=loc[1] - loc[0])[max(loc[1] - loc[0], 0):]
        elif direction == 'diag2':
            flipped_x = np.fliplr(x)
            diag_offset = (18 - loc[1]) - loc[0]
            path = np.diagonal(flipped_x, offset=diag_offset)[max(diag_offset, 0):]

        first_stone = path[path != 0]
        if len(first_stone) > 1 and first_stone[1] == color:
            return False  # Found a matching stone, chain is not dead
    return True  # If we reach here, no matching stones were found in any direction
def is_territory(x, loc):
    territory = 0
    if x[loc] != 0:
        return False
    for direction in ['up', 'down', 'left', 'right']:
        if direction == 'up':
            path = x[loc[0], loc[1]:]
        elif direction == 'down':
            path = np.flip(x[loc[0], :loc[1]])
        elif direction == 'left':
            path = np.flip(x[:loc[0], loc[1]])
        elif direction == 'right':
            path = x[loc[0]:, loc[1]]
        if len(path) == 0 or len(np.argwhere(path != 0)) == 0:
            continue
        else:
            first_stone = path[path != 0][0]
            territory += first_stone
    if np.abs(territory) < 2:
        return 0
    return np.sign(territory)



    territory_map = np.zeros([19, 19], dtype=int)
    for i in range(19):
        for j in range(19):
            if is_territory(x, (i, j)):
                territory_map[i, j] = 1
    return get_territory(x)[1]


def from_bicharacter(bichar):
    # Ensure input is valid (length of 2 and only alphabetic characters)
    if len(bichar) != 2 or not bichar.isalpha():
        raise ValueError("Input must be a two-letter string")

    positions = [(ord(char) - ord('a') + 0) for char in bichar.lower()]

    return tuple(positions)

def to_bicharacter(coord):
    x = chr(coord[0] + 64)
    y = chr(coord[0] + 64)
    return x + y

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
            territory_map[i, j] = int(is_territory(x, (i, j)))
            score += territory_map[i, j]
    return score, territory_map

def from_tuple(tup):
    return chr(tup[0] + 65) + str(tup[1] + 1)


def score_game(x, black_prisoners, white_prisoners,komi=0):

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

def SelfCaptureError(Exception):
    def __init__(self, message="Self-capture detected!"):
        super().__init__(message)


def remove_dead_stones(x, tracker):
    new_x = x.copy()
    dead_stones = np.zeros(x.shape, dtype=int)
    # Detect atari groups and update the mask
    ataris = np.nonzero(detect_atari_groups(x, tracker))
    for i, j in zip(*ataris):
        new_x[i, j] = 0
        dead_stones[i, j] = x[i, j]
    for i in range(19):
        for j in range(19):
            if is_chain_dead(new_x, (i, j)) or not is_chain_alive(new_x, (i, j)):
                dead_stones[i, j] = x[i, j]
                new_x[i, j] = 0
    return new_x, dead_stones
def end_game(x, tracker):
    print('Scoring...')
    y, dead = remove_dead_stones(x, tracker)
    score, territory, winner = score_game(y, 0, 0)
    fig, ax = plot_board(y, territory=territory, fig=None, ax=None)
    fig, ax = plot_board(dead, stone_opacity=0.5, fig=fig, ax=ax)
    ax.imshow(go_board, extent=[0, 24, 0, 18], zorder=-1)
    print('Black score: ' + str(score) + ' points')
    print('Winner: ' + ('Black' if winner == 1 else 'White'))
    ax.savefig('static/board' + str(np.randint(0, 1000)) + '.png')
    plt.show()
    return winner



def load_sgf(sgf):
    game_5 = open(sgf, 'r')
    text = game_5.read()
    pattern = r';([BW])\[([a-z]{2})\]'
    matches = re.findall(pattern, text)
    for i in range(len(matches)):
        matches[i] = (matches[i][0], from_bicharacter(matches[i][1]))
    return matches


def run_from_goban():

    x = np.zeros([19, 19], dtype=int)
    ko_states = set()
    tracker = BoardTracker()
    sgf = os.path.join(static, 'kot.sgf')
    acts = load_sgf(sgf)
    move_count = 0

    for _, loc in acts:
        color = 1 if move_count % 2 == 0 else -1
        move_count += 1
        x, tracker = action(x, loc, color, tracker)
        x, tracker, ko_states = process(x, loc, color, tracker, ko_states)

    end_game(x, tracker)

    plt.show()


if __name__ == '__main__':
    print(sys.version)
    x = np.zeros([19, 19], dtype=int)
    run_from_goban()
