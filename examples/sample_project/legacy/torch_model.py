import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LegacyNet(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        hidden = F.relu(self.fc1(inputs))
        return self.fc2(hidden)


def run_training_epoch():
    model = LegacyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    fake_features = Variable(torch.randn(32, 4))
    fake_labels = Variable(torch.randn(32, 1))

    optimizer.zero_grad()
    outputs = model(fake_features)
    loss = criterion(outputs, fake_labels)
    loss.backward()
    optimizer.step()

    print(f"Legacy PyTorch training loss: {loss.item():.4f}")
