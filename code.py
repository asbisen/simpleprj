import torch
import numpy as np
import matplotlib.pyplot as plt


def py310check():
    import collections
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from collections import MutableMapping  # Removed in Python 3.10+

    import asyncio
    async def example_coroutine():
        print("Running coroutine")

    # Deprecated parameter `loop`
    loop = asyncio.get_event_loop()
    asyncio.run_coroutine_threadsafe(example_coroutine(), loop=loop)

py310check()


# Generate synthetic data
x = np.linspace(-1, 1, 100, dtype=np.float32)
y = 2 * x + 1 + np.random.normal(0, 0.1, size=x.shape).astype(np.float32)

# Convert to PyTorch tensors
x_tensor = torch.from_numpy(x).reshape(-1, 1)
y_tensor = torch.from_numpy(y).reshape(-1, 1)

# Define a simple linear model
model = torch.nn.Sequential(
    torch.nn.Linear(1, 1)
)

from torch.autograd import Variable
x_var = Variable(x_tensor)
y_var = Variable(y_tensor)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(200):
    # Forward pass
    y_pred = model(x_var)
    loss = criterion(y_pred, y_var)

    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict using the trained model
with torch.no_grad():
    y_pred_plot = model(x_var).numpy()

# Plot results
plt.scatter(x, y, label='Data', color='blue')
plt.plot(x, y_pred_plot, label='Prediction', color='red')
plt.title('Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Save plot to a file
plt.savefig('output.png')
plt.show()

