import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


def polynomial_regression(X, y, degree):
    X_poly = np.column_stack([X**i for i in range(1, degree + 1)])
    coefficients = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return coefficients


degree = 2
coefficients = polynomial_regression(X, y, degree)


def calculate_rmse(X, y, coefficients):
    y_pred = X @ coefficients
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return rmse


X_poly = np.column_stack([X**i for i in range(1, degree + 1)])
rmse = calculate_rmse(X_poly, y, coefficients)
print("Root Mean Squared Error (RMSE):", rmse)

X_curve = np.linspace(0, 2, 100).reshape(-1, 1)
X_curve_poly = np.column_stack([X_curve**i for i in range(1, degree + 1)])
y_curve = X_curve_poly @ coefficients

plt.scatter(X, y, label="Data Points")
plt.plot(X_curve, y_curve, color="red", label="Polynomial Regression Curve")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression")
plt.show()

new_data_point = np.array([[1.5]])
new_data_point_poly = np.column_stack(
    [new_data_point**i for i in range(1, degree + 1)]
)
predicted_value = new_data_point_poly @ coefficients
print("Predicted Value for New Data Point:", predicted_value[0, 0])
