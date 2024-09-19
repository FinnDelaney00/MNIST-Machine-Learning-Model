import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data.head()

# Convert the dataset to a NumPy array for easier manipulation
data = np.array(data)
# Get the number of examples (m) and the number of features (n) from the dataset's shape
m, n = data.shape
# Randomly shuffle the dataset to ensure a good mix of training and validation data
np.random.shuffle(data)

# Split the first 1000 examples for the development/validation set
data_dev = data[0:1000].T  # Transpose the data for easier column-wise access
Y_dev = data_dev[0]        # The first row contains the labels
X_dev = data_dev[1:n]      # The remaining rows contain the features
# Normalize the feature values to be in the range [0, 1]
X_dev = X_dev / 255.

# Split the remaining examples for the training set
data_train = data[1000:m].T  # Transpose the data for easier column-wise access
Y_train = data_train[0]      # The first row contains the labels
X_train = data_train[1:n]    # The remaining rows contain the features
# Normalize the feature values to be in the range [0, 1]
X_train = X_train / 255.
# Get the number of training examples
_, m_train = X_train.shape

Y_train

def init_params():
    # Initialize W1 with random values in the range [-0.5, 0.5] with shape (10, 784)
    W1 = np.random.rand(10, 784) - 0.5
    # Initialize b1 with random values in the range [-0.5, 0.5] with shape (10, 1)
    b1 = np.random.rand(10, 1) - 0.5
    # Initialize W2 with random values in the range [-0.5, 0.5] with shape (10, 10)
    W2 = np.random.rand(10, 10) - 0.5
    # Initialize b2 with random values in the range [-0.5, 0.5] with shape (10, 1)
    b2 = np.random.rand(10, 1) - 0.5
    # Return the initialized parameters
    return W1, b1, W2, b2


def ReLU(Z):
    # Apply the ReLU activation function: max(0, Z)
    return np.maximum(Z, 0)


def softmax(Z):
    # Compute the softmax of each column of Z
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    # Compute the linear combination of inputs and weights for the first layer
    Z1 = W1.dot(X) + b1
    # Apply ReLU activation function
    A1 = ReLU(Z1)
    # Compute the linear combination of activations and weights for the second layer
    Z2 = W2.dot(A1) + b2
    # Apply softmax activation function
    A2 = softmax(Z2)
    # Return all computed values
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    # Compute the derivative of ReLU function
    return Z > 0

def one_hot(Y):
    # Initialize a matrix of zeros with shape (number of examples, number of classes)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # Set the appropriate elements to 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose the matrix to match the required shape
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Convert the labels to one-hot encoded matrix
    one_hot_Y = one_hot(Y)
    # Compute the gradient of the loss with respect to Z2
    dZ2 = A2 - one_hot_Y
    # Compute the gradient of the loss with respect to W2
    dW2 = 1 / m * dZ2.dot(A1.T)
    # Compute the gradient of the loss with respect to b2
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    # Compute the gradient of the loss with respect to Z1
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    # Compute the gradient of the loss with respect to W1
    dW1 = 1 / m * dZ1.dot(X.T)
    # Compute the gradient of the loss with respect to b1
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    # Return all computed gradients
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # Update W1 by subtracting the learning rate times the gradient of W1
    W1 = W1 - alpha * dW1
    # Update b1 by subtracting the learning rate times the gradient of b1
    b1 = b1 - alpha * db1
    # Update W2 by subtracting the learning rate times the gradient of W2
    W2 = W2 - alpha * dW2
    # Update b2 by subtracting the learning rate times the gradient of b2
    b2 = b2 - alpha * db2
    # Return the updated parameters
    return W1, b1, W2, b2

def get_predictions(A2):
    # Use np.argmax to find the index of the maximum value in each column of A2.
    # This index corresponds to the predicted class label for each example.
    return np.argmax(A2, axis=0)

def get_accuracy(predicts, Y):
    # Print the predictions and actual labels for debugging purposes.
    print(predicts, Y)
    # Calculate the accuracy as the number of correct predictions divided by the total number of examples.
    return np.sum(predicts == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    # Initialize the weights and biases.
    W1, b1, W2, b2 = init_params()
    # Iterate for the given number of iterations.
    for i in range(iterations):
        # Perform forward propagation to get the activations.
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        # Perform backward propagation to get the gradients.
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        # Update the weights and biases using the calculated gradients.
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # Every 10 iterations, print the current iteration and accuracy.
        if i % 10 == 0:
            print("iteration: ", i)
            print("accuracy: ", get_accuracy(get_predictions(A2), Y))
    # Return the final weights and biases after training.
    return W1, b1, W2, b2

#trains the neural network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

def make_predictions(X, W1, b1, W2, b2):
    # Perform forward propagation using the provided weights and biases.
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    # Get the predictions by converting the output of the softmax layer to class labels.
    predictions = get_predictions(A2)
    # Return the predicted class labels.
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    # Extract the image at the given index from the training data, maintaining its shape.
    current_image = X_train[:, index, None]
    # Make a prediction for the image at the given index using the provided weights and biases.
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    # Get the true label for the image at the given index.
    label = Y_train[index]
    # Print the predicted class label.
    print("Prediction: ", prediction)
    # Print the true class label.
    print("Label: ", label)
    # Reshape the image to its original 28x28 form and scale it back to [0, 255] range.
    current_image = current_image.reshape((28, 28)) * 255
    # Set the color map to grayscale.
    plt.gray()
    # Display the image using matplotlib.
    plt.imshow(current_image, interpolation='nearest')
    # Show the image plot.
    plt.show()

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)

test_prediction(62,W1,b1,W2,b2)
