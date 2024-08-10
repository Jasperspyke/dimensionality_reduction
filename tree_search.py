from baduk import *
import copy
from concurrent.futures import ThreadPoolExecutor

tsc = {
    'get_uct': 0,
    'node init': 0,
    'expand': 0,
    'policy': 0,
    'backpropagate': 0,
    'select_uct': 0,

}
def get_uct(node, c=1, k=5):
    return node.value / (k * node.number) + np.sqrt(np.log(k * node.number) / (k * node.number + 1)) * node.policy * c


def pdf(loc, mean, std):
    x = loc[0]
    y = loc[1]
    distance = np.sqrt((x - mean[0]) ** 2 + (y - mean[1]) ** 2)
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-distance / (2 * std ** 2))
    return y


def scuffed_policy(legal_moves):
    global tsc
    tstart = time.perf_counter()

    mean = (4, 4)
    std = 2
    x = np.ones([19, 19], dtype=int)

    y = np.zeros([19, 19])
    for i in range(19):
        for j in range(19):
            y[i, j] = pdf((i, j), mean, std)
            if not [i, j] in legal_moves:
                y[i, j] = 0
    y = y / np.sum(y)
    tend = time.perf_counter()
    tsc['policy'] += tend - tstart
    return y


class Node:

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


            self.board, self.tracker, self.ko_states = process(self.board, self.loc, self.color, self.tracker, self.ko_states)
            temp = self.tracker.white
            self.tracker.white = self.tracker.black
            self.tracker.black = temp

        self.legal_actions = self.tracker.white.legal_moves if self.color == -1 else self.tracker.black.legal_moves

    def expand(self):
        assert len(self.children) == 0
        global tsc
        tstart = time.perf_counter()
        legal_list = np.argwhere(self.legal_actions)
        if len(legal_list) == 0:
            raise ValueError('No legal moves!')
        count = 0
        children = np.empty(0, dtype=Node)
        policy = scuffed_policy(legal_list)
        for loc in np.argwhere(self.legal_actions):
            count += 1
            new_board = self.board.copy()
            new_board, __ = action(new_board, loc, self.color, self.tracker)
            val = np.random.rand() / 10 + 0.5
            new_state = {'board': new_board, 'loc': tuple(loc), 'color': self.color*-1, 'passed': False,  'number': 1.01, 'policy': policy[loc[0], loc[1]], 'value': val, 'is_leaf': True, 'tracker': None, 'ko_states': None}
            new_state = Node(new_state, self)
            children = np.append(children, new_state)

        self.children = children
        self.is_leaf = False


    def __repr__(self):
        st = 'I am a node named ' + str(id(self)) + '.'
        return st
    def backpropagate(self):
        global tsc
        tstart = time.perf_counter()
        z = self.value
        node = self
        while node.parent is not None:
            node = node.parent
            node.number += 1
            node.value += z
        tsc['backpropagate'] += time.perf_counter() - tstart


     # find the child with the highest UCT value and return it
    def select_uct(self):
        global tsc
        tstart = time.perf_counter()
        uct_values = np.array([get_uct(child) for child in self.children])
        where_max = np.argmax(uct_values)
        tend = time.perf_counter()
        tsc['select_uct'] += tend - tstart
        return self.children[where_max]


def dfs_child_count(root):
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
    depth = 0
    while not node.is_leaf:
        node = node.select_uct()
        depth += 1
    return node

def ascend(node):
    while node.parent is not None:
        node = node.parent
    return node

def count_max_depth(node):
    depth = 0
    while not node.is_leaf:
        node = node.select_uct()
        depth += 1
    return depth

def iteration(node):
    node = descend(node)
    node.expand()
    node = node.select_uct()
    node.backpropagate()
    node = ascend(node)


x = np.zeros([19, 19], dtype=int)
initial_state = {'board': x, 'loc': None, 'color': 1, 'passed': False, 'number': 1, 'policy': 1, 'value': 0.0, 'is_leaf': True, 'tracker': BoardTracker(), 'ko_states': set()}
root = Node(initial_state)
t0 = time.perf_counter()

for i in range(1000):
    print('iteration, ', i)
    iteration(root)




print('max depth is: ', count_max_depth(root))
print('Number of nodes:', dfs_child_count(root))
print('Time information: ', tsc)
print('Baduk time info (for comparison): ', time_consuming_functions)
t2 = time.perf_counter()
print('Time elapsed:', t2 - t0)