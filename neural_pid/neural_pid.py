'''
Author: zhangL2auto
Date: 2024-01-24 14:18:35
LastEditors: 2auto deutschlei47@126.com
LastEditTime: 2024-01-24 14:18:45
'''
import numpy as np
import matplotlib.pyplot as plt

# Step signal
t_step = np.arange(0, 10, 0.01)
u_step = np.ones_like(t_step)

# Sine signal
t_sine = np.arange(0, 10, 0.01)
u_sine = np.sin(t_sine)

# Neural network parameters
input_size = 4
hidden_size = 5
output_size = 3

# Initialize weights
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Initialize biases
b1 = np.zeros(hidden_size)
b2 = np.zeros(output_size)

# Training parameters
learning_rate = 0.01
num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    hidden_layer = np.maximum(0, np.dot(u_step, W1) + b1)
    output_layer = np.dot(hidden_layer, W2) + b2

    # Calculate error
    error = output_layer - u_step

    # Backpropagation
    grad_output = 2 * error
    grad_W2 = np.dot(hidden_layer.T, grad_output)
    grad_b2 = np.sum(grad_output, axis=0)
    grad_hidden = np.dot(grad_output, W2.T)
    grad_hidden[hidden_layer <= 0] = 0
    grad_W1 = np.dot(u_step.T, grad_hidden)
    grad_b1 = np.sum(grad_hidden, axis=0)

    # Update weights and biases
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(t_step, u_step, label='Input')
plt.plot(t_step, output_layer[:, 0], label='Output')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_step, u_step - output_layer[:, 0], label='Error')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_sine, u_sine, label='Input')
plt.plot(t_sine, output_layer[:, 1], label='Output')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t_sine, u_sine - output_layer[:, 1], label='Error')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.show()
