import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(128 * 19 * 19 + 1, 362)
        self.fc_value = nn.Linear(128 * 19 * 19 + 1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, legal_move_array):
        legal_move_array = torch.Tensor(legal_move_array)
        legals = torch.cat([legal_move_array.flatten(), torch.Tensor([1]).view(-1,)], dim=0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 19 * 19)
        x = torch.cat([x, torch.Tensor([0]).view(-1, 1)], dim=1)
        x1 = self.fc_policy(x)
        policy = F.softmax(x1, dim=1).detach()
        policy = (policy.flatten() * legals)/torch.sum(policy)
        value = torch.tanh(self.fc_value(x))
        return policy.numpy(), value.item()



if __name__ == "__main__":
    x = torch.zeros((1, 4, 19, 19), dtype=torch.float32)
    legal_moves = torch.rand(19, 19) < 0.5
    policy_net = PolicyNet()
    policy, value = policy_net(x, legal_moves)


    # <3
