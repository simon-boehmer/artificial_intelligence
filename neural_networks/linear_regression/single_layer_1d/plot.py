import matplotlib.pyplot as plt
import numpy as np


def plot_results(X_train, y_train, X_test, y_test, predictions):
    # ----- Figure Setup -----
    # Create a figure with a high resolution (DPI 150) and preferred size
    plt.figure(figsize=(12, 8), dpi=150)

    # ----- Plot Data Points -----
    # Plot training, testing data, and model predictions
    plt.scatter(X_train, y_train, color="blue", edgecolor="k", label="Training data")
    plt.scatter(X_test, y_test, color="green", edgecolor="k", label="Testing data")
    plt.scatter(
        X_test, predictions.flatten(), color="red", edgecolor="k", label="Predictions"
    )

    # ----- Styling -----
    # Set title, axis labels, and grid with custom font sizes and style
    plt.title("Neural Network Predictions", fontsize=16)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    # ----- Final Adjustments & Display -----
    # Adjust layout for neatness and display the plot
    plt.tight_layout()
    plt.show()
