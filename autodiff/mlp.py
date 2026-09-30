import numpy as np
from autodiff import Node


class MulchParseptron:
    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int,
        activator: callable
    ):
        self.activator = activator
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        layer_sizes = [input_size] + hidden_layers + [output_size]

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(Node(np.random.randn(out_size, in_size)))
            self.biases.append(Node(np.random.randn(out_size, 1)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # hidden
        for i in range(len(self.hidden_layers)):
            # "Input operand 1 has a missmatch in its core dimension 0"を回避するためにescape処理
            x = self.weights[i] @ x + self.biases[i]
            x = self.activator(x)

        # output
        x = self.weights[-1] @ x + self.biases[-1]
        return x

    def parameters(self) -> list[np.ndarray]:
        return self.weights + self.biases
