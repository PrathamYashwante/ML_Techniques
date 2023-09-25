import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "SquareFootage": [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    "NumBedrooms": [3, 3, 2, 4, 2, 3, 4, 3, 2, 3],
    "Price": [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

df = pd.DataFrame(data)

X = df[["SquareFootage", "NumBedrooms"]]
y = df["Price"]

X = np.column_stack((np.ones(len(X)), X))

coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

y_pred = X @ coefficients

rmse = np.sqrt(np.mean((y - y_pred) ** 2))
print("Root Mean Squared Error (RMSE):", rmse)

fig, ax = plt.subplots()
ax.scatter(X[:, 1], y, alpha=0.7)
ax.scatter(X[:, 1], y_pred, color='red', marker='x', alpha=0.7)
ax.set_xlabel('Square Footage')
ax.set_ylabel('Price')
ax.set_title('Multiple Linear Regression')
plt.show()

new_property = np.array([1, 2000, 3])
predicted_price = new_property @ coefficients
print("Predicted Price for New Property:", predicted_price)
