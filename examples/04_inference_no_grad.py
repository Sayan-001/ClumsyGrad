# Problem being solved: running a trained model's forward pass without
# paying the cost of building a computational graph, since inference never
# calls backward().

from clumsygrad import activation as act
from clumsygrad.loss import mse_loss
from clumsygrad.optimizer import Adam
from clumsygrad.random import rand, randn
from clumsygrad.tensor import Tensor, TensorType, no_grad

# Train the model from Example 3
X = rand((100, 1), tensor_type=TensorType.INPUT) * 4 - 2  # range: -2 to 2
y = X**2

w1 = randn((1, 10), tensor_type=TensorType.PARAMETER)
b1 = Tensor([[0.0] * 10], tensor_type=TensorType.PARAMETER)
w2 = randn((10, 1), tensor_type=TensorType.PARAMETER)
b2 = Tensor([0.0], tensor_type=TensorType.PARAMETER)
optimizer = Adam([w1, b1, w2, b2], lr=0.05)

for epoch in range(300):
    optimizer.zero_grad()
    hidden = act.relu(X @ w1 + b1)
    prediction = hidden @ w2 + b2
    loss = mse_loss(prediction, y)
    loss.backward()
    optimizer.step()

# Inference with no_grad()
with no_grad():
    hidden = act.relu(X @ w1 + b1)
    prediction = hidden @ w2 + b2

print(prediction.requires_grad)  # False
print(prediction.data[:3].flatten())
print(y.data[:3].flatten())
