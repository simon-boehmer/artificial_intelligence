import numpy as np
from sklearn.model_selection import train_test_split


def generate_data(n_samples, noise):
    # Generate X and reshape it
    # We reshape to convert a 1D array of samples into a 2D array with one column per sample,
    # which is the expected format for many ML models.
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    # Now compute y so that both have the same shape
    y = 3 * X + np.random.randn(n_samples, 1) * noise
    # 80/20 train/test split
    return train_test_split(X, y, test_size=0.2, random_state=42)
