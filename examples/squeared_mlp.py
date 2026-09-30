"""
Example of approximating the y=x^2 function using a neural network with the MLP.
"""

import numpy as np
from matplotlib import pyplot as plt

from autodiff import Node
from autodiff.mlp import MulchParseptron
from optim.svg import SVG


def tanh(x: Node) -> Node:
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())


np.random.seed(0)

model = MulchParseptron(
    input_size=1,
    hidden_layers=[16, 16],
    output_size=1,
    activator=tanh,
)

x = np.linspace(-1, 1, 100).reshape(1, -1)
y = x**2
loss_history = []
optim = SVG(learning_rate=0.01)

for i in range(30):
    pred = model.forward(x)
    loss = ((pred - y) ** 2).mean()

    loss_history.append(float(loss.data))

    loss.backward()

    params = model.parameters()
    optim.step(params, [param.grad for param in params])
    if i % 10 == 0:
        print(f"Epoch {i}: loss = {loss.data}")

fig, ax = plt.subplots()
ax.set_title("Loss over Epochs")
ax.plot(loss_history, label="loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.show()
