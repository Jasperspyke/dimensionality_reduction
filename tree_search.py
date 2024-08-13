import copy
from baduk import *
from model import *

# Time statistics collector
tsc = {
    'get_uct': 0,
    'node init': 0,
    'expand': 0,
    'policy': 0,
    'backpropagate': 0,
    'select_uct': 0,
}


def get_uct(node):
    """Calculate the UCT (Upper Confidence Bound for Trees) value."""
    explore = np.sqrt(np.log(node.parent.number) / node.number)
    exploit = node.value / node.number
    return explore, exploit


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

    def expand(self):
        """Expand the node by adding children for each legal move."""
        assert len(self.children) == 0
        global tsc
        tstart = time.perf_counter()
        legal_list = np.argwhere(self.legal_actions)
        if len(legal_list) == 0:
            raise ValueError('No legal moves!')
        count = 0
        children = np.empty(0, dtype=Node)
        x = self.board
        x_white = x == -1
        x_black = x == 1
        x_empty = x == 0
        x = np.stack((x_white, x_black, x_empty), axis=0, dtype=float)
        x = torch.tensor(x, dtype=torch.float32).clone().detach().unsqueeze(0)
        policy, value = policy_net(x)
        policy = policy.detach().numpy()
        value = value.detach().numpy()
    #    value = np.random.rand()

        for loc in np.argwhere(self.legal_actions):
            count += 1
            new_board = self.board.copy()
            new_board, __ = action(new_board, loc, self.color, self.tracker)
            new_state = {
                'board': new_board,
                'loc': tuple(loc),
                'color': self.color * -1,
                'passed': False,
                'number': self.number + 1,
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
        assert len(self.children) == len(legal_list)

    def __repr__(self):
        return f'I am a node named {id(self)}.'

    def backpropagate(self):
        """Propagate the results up the tree."""
        global tsc
        tstart = time.perf_counter()
        z = self.value
        node = self
        while node.parent is not None:
            node = node.parent
            node.number += 1
            node.value += z
        tsc['backpropagate'] += time.perf_counter() - tstart

    def select_uct(self):
        """Select the child with the highest UCT value."""
        global tsc
        tstart = time.perf_counter()
        uct_values = []
        c = 0.50
        for child in self.children:
            explore, exploit = get_uct(child)
            uct_values.append([c * explore + exploit])
        uct_values = np.array(uct_values)
        where_max = np.argmax(uct_values)
        tend = time.perf_counter()
        tsc['select_uct'] += tend - tstart
        return self.children[where_max]


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


def descend(node):
    """Descend the tree to a leaf node."""
    depth = 0
    while not node.is_leaf:
        node = node.select_uct()
        depth += 1
    return node


def ascend(node):
    """Ascend the tree to the root."""
    while node.parent is not None:
        node = node.parent
    return node


def count_max_depth(node):
    """Count the maximum depth of the tree."""
    depth = 0
    while not node.is_leaf:
        node = node.select_uct()
        depth += 1
    return depth


def iteration(node):
    """Perform one iteration of the MCTS algorithm."""
    node = descend(node)
    node.expand()
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
    'number': 1,
    'policy': 1,
    'value': 0.0,
    'is_leaf': True,
    'tracker': BoardTracker(),
    'ko_states': set(),
}
root = Node(initial_state)
t0 = time.perf_counter()
policy_net = PolicyNet()

for i in range(1000):
    print('iteration:', i)
    iteration(root)

# Output the results
print('max depth is: ', count_max_depth(root))
print('Number of nodes:', dfs_child_count(root))
print('Time information: ', tsc)
print('Baduk time info (for comparison): ', time_consuming_functions)
t2 = time.perf_counter()
print('Time elapsed:', t2 - t0)
