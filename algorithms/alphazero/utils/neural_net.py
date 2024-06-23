import torch.nn as nn
import torch.nn.functional as F
from game_logic import ALL_FIELDS_SIZE


class Net(nn.Module):
    def __init__(self, num_hidden, device):
        super().__init__()
        self.device = device
        self.iterations_trained = 0

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)  # Dropout layer
        )

        self.sharedConv = nn.Sequential(
            nn.Conv2d(num_hidden, 2 * num_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Dropout layer
            nn.Conv2d(2 * num_hidden, 2 * num_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2) , # Dropout layer
            nn.Conv2d(2 * num_hidden, 2 * num_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)  # Dropout layer            
        )

        self.policyHead = nn.Sequential(            
            nn.Conv2d(2 * num_hidden, 1 * num_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * num_hidden * ALL_FIELDS_SIZE, ALL_FIELDS_SIZE)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(2 * num_hidden, 1 * num_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * num_hidden * ALL_FIELDS_SIZE, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        shared_out = self.sharedConv(x)
        policy = self.policyHead(shared_out)
        value = self.valueHead(shared_out)
        return policy, value

# class ResNet(nn.Module):
#     def __init__(self, num_resBlocks, num_hidden, device):
#         super().__init__()
#         self.device = device
#         self.iterations_trained = 0
#
#         self.startBlock = nn.Sequential(
#             nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU()
#         )
#
#         self.backBone = nn.ModuleList(
#             [ResBlock(num_hidden) for i in range(num_resBlocks)]
#         )
#
#         self.policyHead = nn.Sequential(
#             nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, ALL_FIELDS_SIZE)
#         )
#
#         self.valueHead = nn.Sequential(
#             nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
#             nn.BatchNorm2d(3),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, 1),
#             nn.Tanh()
#         )
#
#         self.to(device)
#
#     def forward(self, x):
#         x = self.startBlock(x)
#         for resBlock in self.backBone:
#             x = resBlock(x)
#         policy = self.policyHead(x)
#         value = self.valueHead(x)
#         return policy, value
#
#
# class ResBlock(nn.Module):
#     def __init__(self, num_hidden):
#         super().__init__()
#         self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(num_hidden)
#         self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(num_hidden)
#
#     def forward(self, x):
#         residual = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += residual
#         x = F.relu(x)
#         return x


