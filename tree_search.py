from baduk import *
from inspect import stack
import json

def get_uct(node, c=5000):
    return node.value / node.number + np.sqrt(np.log(node.number) / node.number + 1) * node.policy * c


def pdf(loc, mean, std):
    x = loc[0]
    y = loc[1]
    distance = np.sqrt((x - mean[0]) ** 2 + (y - mean[1]) ** 2)
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-distance / (2 * std ** 2))
    return y

def scuffed_policy(legal_moves):

    mean = (4, 4)
    std = 3
    x = np.ones([19, 19], dtype=int)

    y = np.zeros([19, 19])
    for i in range(19):
        for j in range(19):
            y[i, j] = pdf((i, j), mean, std)
            if not [i, j] in legal_moves:
                y[i, j] = 0
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
        self.number = state['number']
        # information about past expansions goes here
        self.children = np.empty(0, dtype=Node)
        self.policy = state['policy']
        self.value = state['value']
        self.is_leaf = state['is_leaf']


    # expand a selected node by adding a child node for each legal action
    def expand(self):
        assert self.is_leaf

        policy = scuffed_policy(self.legal_actions)
        count = 0
        children = np.empty(0, dtype=Node)
        for loc in self.legal_actions:
            count += 1

            new_board = self.board.copy()
            new_board = action(new_board, loc, self.color, self.number)
            assert not np.all(new_board == self.board)
            if count == 6666:
                val = 1
            else:
                val = np.random.rand() / 10 + 0.5
            new_state = {'board': new_board, 'color': self.color*-1, 'passed': False,  'number': 1.01, 'policy': policy[loc[0], loc[1]], 'value': val, 'is_leaf': True}

            new_state = Node(new_state, self)
            children = np.append(children, new_state)
        self.children = children
        self.is_leaf = False
    def __repr__(self):
        st = 'I am a node named' + str(id(self)) + '.'
    def backpropagate(self):
        z = self.value
        node = self
        while node.parent is not None:
            node = node.parent
            node.number += 1
            node.value += z


     # find the child with the highest UCT value and return it
    def select_uct(self):
        assert not self.is_leaf
        uct_values = np.empty(0)
        for child in self.children:
            uct_values = np.append(uct_values, get_uct(child))

        where_max = np.argmax(uct_values)
        uct_values[where_max] = -np.inf
        second_where_max = np.argmax(uct_values)
        c1 = self.children[where_max]
        c2 = self.children[second_where_max]
        if self == root:
            print('UCT values:', get_uct(c1), get_uct(c2))
            print('c1 exploitation term:', c1.value/c1.number)
            print('c2 exploitation term:', c2.value/c2.number)
            print('c1 exploration term:', np.sqrt(np.log(c1.number)/c1.number+1) * c1.policy * 5000)
            print('c2 exploration term:', np.sqrt(np.log(c2.number)/c2.number+1) * c2.policy * 5000)
            if c2.number > 1.01:
                print('boom!!!')
        return self.children[where_max]



def dfs_child_count(root):
    count = 1
    stack = [root]
    while stack:
        node = stack.pop()
        count += len(node.children)
        for child in node.children:
            stack.append(child)

    return count
def descend(node):
    while not node.is_leaf:
        node = node.select_uct()
    return node

def iteration(node):
    node = descend(node)
    node.expand()
    node = node.select_uct()
    node.backpropagate()





initial_state = {'board': x, 'color': 1, 'passed': False, 'number': 1, 'policy': 1, 'value': 0.0, 'is_leaf': True}
root = Node(initial_state)
t_ = time.perf_counter()

for i in range(1000):
    iteration(root)

function_times = time_consuming_functions
print('Time consuming functions:', function_times)
print('Number of nodes:', dfs_child_count(root))
print('Saving as functimes.json...')
with open('functimes.json', 'w') as f:
    json.dump(function_times, f)

t__ = time.perf_counter()
print('Time elapsed:', t__ - t_)