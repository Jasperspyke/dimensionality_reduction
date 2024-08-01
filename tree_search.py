from baduk import *


def get_uct(node, c=1):
    return node.wins/node.number + np.sqrt(node.number/node.number+1)

def pdf(loc, mean, std):
    x = loc[0]
    y = loc[1]
    distance = np.sqrt((x - mean[0]) ** 2 + (y - mean[1]) ** 2)
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-distance / (2 * std ** 2))
    return y

def scuffed_policy(legal_moves):

    mean = (4, 4)
    std = 1
    x = np.ones([19, 19], dtype=int)

    y = np.zeros([19, 19])
    for i in range(19):
        for j in range(19):
            y[i, j] = pdf((i, j), mean, std)

    y = y * legal_moves
    y = y / np.sum(y)
    return y

class Node:

    def __init__(self, state, parent=None):
        self.board = state['board']
        self.parent = parent

        self.color = state['color']
        self.legal_actions = np.argwhere(legal_move_array(self.board, self.color))
        if len(self.legal_actions) == 0:
            sys.exit('The game is over!')
        self.passed = state['passed']
        self.wins = state['wins']
        self.number = state['number']
        # information about past expansions goes here
        self.children = np.empty(0, dtype=Node)
        self.policy = state['policy']
        state.value = 0

    def expand_leaf(self):
        policy = scuffed_policy(self.legal_actions)
        for loc in self.legal_actions:

            new_board = self.board.copy()
            new_board = action(new_board, loc, self.color, self.number)
            new_state = {'board': new_board, 'color': self.color*-1, 'passed': False, 'wins': 0, 'number': 0.01, 'policy': policy[loc[0], loc[1]], 'value': 0.333}

            new_state = Node(new_state, self)
            self.children.append(new_state)

    def leaf2expand(self):
        if len(self.children) == 0:
            return self
        else:
            uct_values = get_uct(self.children)
            where_max = np.argmax(uct_values)
            return self.children[where_max]

    def __repr__(self, level=0):
        # Create a local variable y to hold the modified board
        y = self.board.copy()
        num_rows, num_cols = y.shape
        y_str = y.astype(str)
        col_index = np.arange(num_cols).astype(str)
        header_row = np.insert(col_index, 0, '')
        y_with_col_index = np.vstack([header_row, np.column_stack([np.arange(num_rows).astype(str), y_str])])
        df = pd.DataFrame(y_with_col_index)
        s1 = df.to_string(index=False, header=False)
        s2 = '\n___________________________________________________________'

        return s1 + s2


def rollout(node):
    board = node.board
    color = node.color
    board = np.zeros([19, 19], dtype=np.int8)
    ko_states = set()
    move_count = 0
    game_over = False
    while move_count < 100:
        if move_count % 2 == 0:
            color = 1
        else:
            color = -1
        moves = np.argwhere(legal_move_array(board, color))
        moves = list(moves)

        moves.append('pass')
        num_moves = len(moves)
        if len(moves) == 0:
            print('no legal moves on move:', move_count)

            move_count = 4004

        rn = np.random.randint(0, num_moves)
        loc = moves[rn]

        #problem with loc and moves

        if type(loc) == str and node.passed:
            print('pass on move:', move_count)

            move_count = 9999
        elif type(loc) == str:
            node.passed = True
            continue
        else:
            y = action(board, loc, color, move_count)
            if str(y) in ko_states:
                raise KoError
            ko_states.add(str(y))
            board = y
            node.passed = False
        move_count += 1

    return board

initial_state = {'board': x, 'color': 1, 'passed': False, 'wins': 0, 'number': 1}
root = Node(initial_state)
root.expand((1, 1))
print(root.children[0])
