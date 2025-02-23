from value import Value
from neuron import MLP

# Training data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, -1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Initialize the network
n = MLP(3, [4, 4, 1])

# Training loop
for k in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), start=Value(0))

    # Backward pass
    for p in n.parameters():
        p.grad = 0.0  # Zero out gradients before backward pass
    loss.backward()

    # Update parameters
    for p in n.parameters():
        p.data -= 0.01 * p.grad

    print(f"Epoch {k}: Loss = {loss.data}")