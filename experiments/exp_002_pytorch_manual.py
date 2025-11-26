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

x = torch.Tensor([
    [
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 1.1, 1.2, 1.3],
        [1.4, 1.5, 1.6, 1.7],
    ],
    [
        [11., 22., 33., 44.],
        [55., 66., 77., 88.],
        [99., 11.11, 11.22, 11.33],
        [11.44, 11.55, 11.66, 11.77],
    ],
    [
        [111., 222., 333., 444.],
        [555., 666., 777., 888.],
        [999., 111.111, 111.222, 111.333],
        [111.444, 111.555, 111.666, 111.777],
    ],
])
print(f"x:\n{x}")
print(x.shape)

convo2d = nn.Conv2d(3, 6, 2, 1)(x)
print(f"convo2d:\n{convo2d}")
print(convo2d.shape)

max_pool = nn.MaxPool2d(2)(x)
print(f"max_pool2d:\n{max_pool}")
print(max_pool.shape)

transp = nn.ConvTranspose2d(6,3,2,1)(convo2d.unsqueeze(0))
print(f"conv_transpode:\n{transp}")
print(transp.shape)

out_conv = nn.Conv2d(3, 9, 1)(transp)
print(f"out conv:\n{out_conv}")
print(out_conv.shape)


weights = F.softmax(out_conv, dim=1)
print(f"weights: {weights}")
print(weights.shape)

view = weights.view(1, 9, 16)
print(f"weights.view: {view }")
print(view .shape)

patches = F.unfold(input=x, kernel_size=3, padding=1)
print(f"patches:\n{patches}")
print(patches.shape)


x = torch.arange(start=1, end=28, dtype=torch.float32).view(1, 3, 3, 3)
print(f"x:\n{x}\n{x.shape}")

raw_weights = torch.arange(start=1, end=82, dtype=torch.float32).view(1, 9, 3, 3)
weights = raw_weights#F.softmax(raw_weights, dim=1)
print(f"weights:\n{weights}\n{weights.shape}")
weights_flat = weights.view(1, 9, 9)
print(f"weights_flat:\n{weights_flat}\n{weights_flat.shape}")
#weights_rgb = weights_flat.repeat(1, 3, 1).view(1, 3, )
#print(f"weights_rgb:\n{weights_rgb}\n{weights_rgb.shape}")

patches = F.unfold(input=x, kernel_size=3, padding=1)
print(f"patches:\n{patches}\n{patches.shape}")

B, CK2, P = patches.shape  # P = H*W = 9
C = 3
K2 = 9

patches_rgba = patches.view(1, 3, 9, 9)
print(f"patches_rgba:\n{patches_rgba}\n{patches_rgba.shape}")
out = (weights_flat * patches_rgba).sum(dim=2).view(1, 3, 3, 3)
print(f"out:\n{out}\n{out.shape}")


# weights_flat:
# tensor([[[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
#          [10., 11., 12., 13., 14., 15., 16., 17., 18.],
#          [19., 20., 21., 22., 23., 24., 25., 26., 27.],
#          [28., 29., 30., 31., 32., 33., 34., 35., 36.],
#          [37., 38., 39., 40., 41., 42., 43., 44., 45.],
#          [46., 47., 48., 49., 50., 51., 52., 53., 54.],
#          [55., 56., 57., 58., 59., 60., 61., 62., 63.],
#          [64., 65., 66., 67., 68., 69., 70., 71., 72.],
#          [73., 74., 75., 76., 77., 78., 79., 80., 81.]]])
# 
# patches_rgba:
# tensor([[[[ 0.,  0.,  0.,  0.,  1.,  2.,  0.,  4.,  5.],
#           [ 0.,  0.,  0.,  1.,  2.,  3.,  4.,  5.,  6.],
#           [ 0.,  0.,  0.,  2.,  3.,  0.,  5.,  6.,  0.],
#           [ 0.,  1.,  2.,  0.,  4.,  5.,  0.,  7.,  8.],
#           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
#           [ 2.,  3.,  0.,  5.,  6.,  0.,  8.,  9.,  0.],
#           [ 0.,  4.,  5.,  0.,  7.,  8.,  0.,  0.,  0.],
#           [ 4.,  5.,  6.,  7.,  8.,  9.,  0.,  0.,  0.],
#           [ 5.,  6.,  0.,  8.,  9.,  0.,  0.,  0.,  0.]],
# 
#          [[ 0.,  0.,  0.,  0., 10., 11.,  0., 13., 14.],
#           [ 0.,  0.,  0., 10., 11., 12., 13., 14., 15.],
#           [ 0.,  0.,  0., 11., 12.,  0., 14., 15.,  0.],
#           [ 0., 10., 11.,  0., 13., 14.,  0., 16., 17.],
#           [10., 11., 12., 13., 14., 15., 16., 17., 18.],
#           [11., 12.,  0., 14., 15.,  0., 17., 18.,  0.],
#           [ 0., 13., 14.,  0., 16., 17.,  0.,  0.,  0.],
#           [13., 14., 15., 16., 17., 18.,  0.,  0.,  0.],
#           [14., 15.,  0., 17., 18.,  0.,  0.,  0.,  0.]],
# 
#          [[ 0.,  0.,  0.,  0., 19., 20.,  0., 22., 23.],
#           [ 0.,  0.,  0., 19., 20., 21., 22., 23., 24.],
#           [ 0.,  0.,  0., 20., 21.,  0., 23., 24.,  0.],
#           [ 0., 19., 20.,  0., 22., 23.,  0., 25., 26.],
#           [19., 20., 21., 22., 23., 24., 25., 26., 27.],
#           [20., 21.,  0., 23., 24.,  0., 26., 27.,  0.],
#           [ 0., 22., 23.,  0., 25., 26.,  0.,  0.,  0.],
#           [22., 23., 24., 25., 26., 27.,  0.,  0.,  0.],
#           [23., 24.,  0., 26., 27.,  0.,  0.,  0.,  0.]]]])
# 
# out:
# tensor([[[ 750., 1239.,  858., 1539., 2385., 1575.,  906., 1347.,  846.],
#          [2730., 4020., 2586., 3942., 5706., 3600., 2130., 2994., 1818.],
#          [4710., 6801., 4314., 6345., 9027., 5625., 3354., 4641., 2790.]]])
