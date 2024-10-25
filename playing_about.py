import matplotlib.pyplot as plt
import numpy as np

# Generate x values from -1 to 1
x = np.linspace(-1, 1, 100)

# Create y values (y = x)
y = x

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='y = x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of y = x')
plt.legend()
plt.grid(True)

# Set axis limits
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()

