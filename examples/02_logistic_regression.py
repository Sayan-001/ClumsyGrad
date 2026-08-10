# Problem being solved: binary classification of points by which side of a
# line they fall on.

from clumsygrad import activation as act
from clumsygrad.loss import mse_loss
from clumsygrad.optimizer import Adam
from clumsygrad.random import randn
from clumsygrad.tensor import Tensor, TensorType

# Points labeled by which side of a line they fall on
X = randn((100, 2), tensor_type=TensorType.INPUT) * 2
true_W = Tensor([[4.0], [4.0]])
target = act.sigmoid(X @ true_W)

W = Tensor([[0.0], [0.0]], tensor_type=TensorType.PARAMETER)
b = Tensor([0.0], tensor_type=TensorType.PARAMETER)
optimizer = Adam([W, b], lr=0.1)

for epoch in range(150):
    optimizer.zero_grad()
    prediction = act.sigmoid(X @ W + b)
    loss = mse_loss(prediction, target)
    loss.backward()
    optimizer.step()

print(f"Loss: {loss.data:.4f}")
