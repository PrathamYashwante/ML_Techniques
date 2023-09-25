import numpy as np
import matplotlib.pyplot as plt

square_footage = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
price = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

def simple_linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - (slope * x_mean)
    return slope, intercept

slope, intercept = simple_linear_regression(square_footage, price)

new_square_footage = 2000
predicted_price = (slope * new_square_footage) + intercept
print("Predicted Price:", predicted_price)

plt.scatter(square_footage, price, label='Data Points')
regression_line = slope * square_footage + intercept
plt.plot(square_footage, regression_line, color='red', label='Regression Line')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()
