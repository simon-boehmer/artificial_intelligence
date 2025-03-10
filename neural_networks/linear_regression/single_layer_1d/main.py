from data import generate_data
from model import create_model
from plot import plot_results

# Generate dummy data
X_train, X_test, y_train, y_test = generate_data(n_samples=200, noise=1.0)

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=50, verbose=1)

# Predict on test data
predictions = model.predict(X_test)

# Plot the results
plot_results(X_train, y_train, X_test, y_test, predictions)
