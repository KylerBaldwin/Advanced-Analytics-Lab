import numpy as np

# Arrange your data as 4 rows x 3 columns
X = np.array([
    [9, 6, 12],
    [11, 7, 22],
    [8, 9, 25],
    [13, 5, 21]
], dtype=float)

# Center the data (deviation matrix)
X_d = X - X.mean(axis=0)

# Compute SSCP
SSCP = X_d.T @ X_d
VARCOV = SSCP / (X.shape[0] - 1)

print("Deviation matrix (X_d):\n", X_d)
print("\nSSCP matrix:\n", SSCP)
print("\nVariance-Covariance matrix:\n", np.cov(X))
