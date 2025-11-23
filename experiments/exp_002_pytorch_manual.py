import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


x = torch.tensor([
    [
        [0.5, 0.3, 0.1,],
        [0.5, 0.3, 0.1,],
        [0.5, 0.3, 0.1,],
    ],
    [
        [0.1, 0.3, 0.2,],
        [0.2, 0.1, 0.3,],
        [0.3, 0.2, 0.2,],
    ],
    [
        [0.1, 0.3, 0.2,],
        [0.2, 0.1, 0.3,],
        [0.1, 0.1, 0.1,],
    ]
])

target = torch.tensor([[[0.3]], [[0.12]], [[0.12]], [[0.12]], [[0.12]], [[0.12]]])

print(f"x.shape {x.shape}, target.shape {target.shape}")

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 )

print(f"Initial weight model0[0]:\n{model[0].weight.data}")
print(f"Initial bias model0[0]:\n{model[0].bias.data}")

# Forward pass
y_pred = model(x)
print(y_pred)

# Calculate loss
loss = loss_fn(y_pred, target)
print(f"loss: {loss.item()}")

# Backwards: compute gradients
optimizer.zero_grad()
loss.backward() # autograd

print(f"Grad weight, step 0: {model[0].weight.grad}")
print(f"Grad bias, step 0:  {model[0].bias.grad}")

# Update weights
optimizer.step()

print(f"Grad weight, step 1: {model[0].weight.data}")
print(f"Grad bias, step 1: {model[0].bias.data}")

# ax, fig = plt.subplots(1, 2)
# fig[0].imshow(x.cpu().detach().numpy())
# fig[1].imshow(y_pred.cpu().detach().numpy())
# plt.show()

# Model 1

model1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=6, out_channels=6, kernel_size=2, stride=1),
    nn.ReLU(inplace=True),
)

# Forward pass
print(x)
y_pred = model1(x)
print(y_pred)

model2 = nn.Sequential(
    nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=2, stride=2)
)

y_pred1 = model2(y_pred)
print(y_pred1)

model3 = nn.Sequential(
    nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=2, stride=2)
)

y_pred2 = model3(y_pred1)
print(y_pred2)
