import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import time
import sys
print(sys.version)

print(torch.__version__)


x = np.zeros(shape=(19, 19), dtype=float)
x_white = x == -1
x_black = x == 1
x_empty = x == 0
x = np.stack((x_white, x_black, x_empty), axis=0, dtype=float)
x = torch.tensor(x).unsqueeze(0).float()
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(128 * 19 * 19, 361)
        self.fc_value = nn.Linear(128 * 19 * 19, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 19 * 19)
        policy = F.softmax(self.fc_policy(x), dim=1)
        policy = policy.view(19, 19)
        value = torch.tanh(self.fc_value(x))
        return policy, value



if __name__ == "__main__":
    policy_net = PolicyNet()
    t1 = time.perf_counter()
    x = x.clone().detach()
    print(x)
    print(x.size())
    sys.exit()
    y = policy_net(x)
    t2 = time.perf_counter()

    print('Time:', t2 - t1)
    print('Policy:', y[0])
