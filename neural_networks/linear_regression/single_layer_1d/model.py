from keras import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    # A single Dense layer with 1 neuron and a linear activation function
    model = Sequential([Dense(1, activation="linear")])
    # Use stochastic gradient descent as an optimizer and mean squared error as a loss function
    model.compile(optimizer="sgd", loss="mse")
    # Return the model
    return model
