import copy
from baduk import *
from model import *
import pandas as pd
import matplotlib.pyplot as plt
import weiqi as wq

class MovesDepletedError(Exception):
    pass


# Time statistics collector
tsc = {
    'calculate_uct': 0,
    'node init': 0,
    'expand': 0,
    'policy': 0,
    'backpropagate': 0,
    'select_uct': 0,
}
def calculate_uct(node, c=1):
    """Calculate the UCT (Upper Confidence Bound for Trees) value from the perspective of the parent node."""

    p = node.policy
    explore = node.value / node.number
    exploit = p * c * np.sqrt(node.parent.number) / (1 + node.number)
    return explore + exploit + np.random.rand() * 1e-6


class Node:
    """Node in a Monte Carlo Tree Search (MCTS) tree."""

    def __init__(self, state, parent=None):
        global tsc
        tstart = time.perf_counter()
        self.board = state['board']
        self.ko_states = set() if parent is None else parent.ko_states
        self.parent = parent
        self.color = state['color']
        self.passed = state['passed']
        self.number = state['number']
        self.children = np.empty(0, dtype=Node)
        self.policy = state['policy']
        self.value = state['value']
        self.is_leaf = state['is_leaf']
        self.loc = state['loc']
        self.tracker = state['tracker']
        self.ko_states = state['ko_states']

        if self.tracker is None:
            tend = time.perf_counter()
            tsc['node init'] += tend - tstart
            self.ko_states = parent.ko_states

            tracky = BoardTracker()
            tracky.white = copy.deepcopy(parent.tracker.white)
            tracky.black = copy.deepcopy(parent.tracker.black)

            self.tracker = BoardTracker()
            self.tracker.white = tracky.white
            self.tracker.black = tracky.black

            self.board, self.tracker, self.ko_states = process(
                self.board, self.loc, self.color, self.tracker, self.ko_states
            )
            temp = self.tracker.white
            self.tracker.white = self.tracker.black
            self.tracker.black = temp

        self.legal_actions = (
            self.tracker.white.legal_moves
            if self.color == -1
            else self.tracker.black.legal_moves
        )

    def expand(self, policy_net):
        """Expand the node by adding children for each legal move."""
        assert len(self.children) == 0
        global tsc
        tstart = time.perf_counter()
        policy, value = self.legal_policy(policy_net)
        children = np.empty(0, dtype=Node)
        for loc in np.argwhere(self.legal_actions):
            new_board = self.board.copy()
            new_board, __ = action(new_board, loc, self.color, self.tracker)
            new_state = {
                'board': new_board,
                'loc': tuple(loc),
                'color': self.color * -1,
                'passed': False,
                'number': self.number + 1 + self.number+0.01 ** 2,
                'policy': policy[loc[0], loc[1]],
                'value': value,
                'is_leaf': True,
                'tracker': None,
                'ko_states': None,
            }
            new_state = Node(new_state, self)
            children = np.append(children, new_state)
        tend = time.perf_counter()
        tsc['expand'] += tend - tstart
        self.children = children
        self.is_leaf = False


    def legal_policy(self, policy_net):
        color = self.color
        legal_list = np.argwhere(self.legal_actions)
        if len(legal_list) == 0:
            raise MovesDepletedError
        x = self.board
        x_white = x == -1
        x_black = x == 1
        x_empty = x == 0
        player = np.ones((19, 19)) * color
        x = np.stack((x_white, x_black, x_empty, player), axis=0, dtype=float)
        x = torch.tensor(x, dtype=torch.float32).clone().detach().unsqueeze(0)
        policy, value = policy_net(x, self.legal_actions)
        return policy, value

    def __repr__(self):
        return f'I am a node named {id(self)}.'

    def backpropagate(self):
        """Propagate the results up the tree."""
        global tsc
        tstart = time.perf_counter()
        z = self.value
        node = self
        while node.parent is not root:
            node = node.parent
            node.number += 1
            node.value += z
        assert node.parent is root
        node.value -= 0.05
        tsc['backpropagate'] += time.perf_counter() - tstart

    def select_uct(self):
        """Select the child with the highest UCT value."""
        global tsc
        tstart = time.perf_counter()
        uct_values = []
        for child in self.children:
            uct = calculate_uct(child)
            uct_values.append(uct)
        uct_values = np.array(uct_values)
   #     print('uct values are: ', uct_values)
        where_max = np.argmax(uct_values)
    #    print('where max is: ', where_max)
        tend = time.perf_counter()
        tsc['select_uct'] += tend - tstart
        return self.children[where_max]


def descend(node):
    """Descend the tree to a leaf node."""
    depth = 1
    while not node.is_leaf:
        node = node.select_uct()
        depth += 1
    return node


def ascend(node):
    """Ascend the tree to the root."""
    while node.parent is not None:
        node = node.parent
    return node

def dfs_child_count(root):
    """Perform a depth-first search to count all nodes."""
    visited = set()
    count = 0
    stack = [root]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            count += 1
            for child in node.children:
                if child not in visited:
                    stack.append(child)

    return count


 # Choose move by running MCTS from the root node.
def take_action(node, policy_net, num_iterations=100):
    assert node.is_leaf
    for i in range(num_iterations):
        iteration(node, policy_net)
    win_rates = []
    for i in (node.children):
        win_rate = i.value/i.number + np.random.rand()*0.001 if i.number > 0 else 0
        win_rates.append(win_rate)
    highest_win = np.max(win_rates)
    best_move = node.children[win_rates.index(highest_win)]
    probability, value = best_move.legal_policy(policy_net)
    outcome = np.zeros((19, 19))
    outcome[best_move.loc] = 1
    winner = 1
    color = node.color

    # Node instance, predicted move probability, move event vector (19x19)
    return best_move, {'probability': probability, 'outcome': outcome, 'value': value, 'winner': winner, 'color': color}



def iteration(node, policy_net):
    """Perform one iteration of the MCTS algorithm."""
    if not node.legal_actions.sum() > 0:
        print('node is: ', node)
        print('legal actions are: ', node.legal_actions)
        sys.exit(1)
    node = descend(node)
    node.expand(policy_net=policy_net)
    node = node.select_uct()
    node.backpropagate()
    node = ascend(node)


# Initialize the root node and start the iterations
x = np.zeros([19, 19], dtype=int)
initial_state = {
    'board': x,
    'loc': None,
    'color': 1,
    'passed': False,
    'number': 0,
    'policy': 1,
    'value': 0.0,
    'is_leaf': True,
    'tracker': BoardTracker(),
    'ko_states': set(),
}

def n_iterations(n, node, policy_net):
    for i in range(n):
        if i % 100 == 0:
            print('iteration:', i)
            print('board state is: ', node.board)
        iteration(node, policy_net)

def playout(node):

    results = pd.DataFrame(columns=['probability', 'outcome', 'value', 'winner', 'color'])
    policy_net = PolicyNet()
    w = node
    count = 0
    while True:
        try:
            print('Playout iteration: ', count)
            x, results.loc[-1] = take_action(w, policy_net)
            assert not np.all(x.board == np.zeros_like(w.board))
            x.value = 0
            x.number = 0
            x.passed = False
            x.is_leaf = True
            x.children = np.empty(0, dtype=Node)
            w = x
            count += 1
        except MovesDepletedError:
            print('Playout successfully terminated!')
            results.columns[-1] *= wq.end_game(x.board, x.tracker)
            print('Results are: ', results)
            pd.to_csv('static/results.csv')
            sys.exit(1)
            break

    return node


if __name__ == "__main__":
    print('Python version is: ', sys.version)
    root = Node(initial_state)
    new_state = playout(root)
    res = new_state.board
    assert not np.all(res == 0)
    np.save('static/board.npy', res)
    wq.end_game(res)
