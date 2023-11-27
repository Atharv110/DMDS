import numpy as np

# Given observations
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12, 16, 18])

# Calculating the coefficients (b0 and b1) for simple linear regression
n = len(x)

x_mean = np.mean(x)
y_mean = np.mean(y)

b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b0 = y_mean - b1 * x_mean

print(f"Estimated coefficient b0 (intercept): {b0}")
print(f"Estimated coefficient b1 (slope): {b1}")
