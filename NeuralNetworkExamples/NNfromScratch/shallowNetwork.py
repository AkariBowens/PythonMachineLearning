import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/ (1 + np.exp(-z))

z = np.linspace(-8, 8, 1000)
y = sigmoid(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.title('logistic')
plt.grid()
plt.show()

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

z = np.linspace(-8, 8, 1000)
y = tanh(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.title('tanh')
plt.grid()
plt.show()

def relu(z):
    return np.maximum(np.zeros_like(z), z)

z = np.linspace(-8, 8, 1000)
y = relu(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.title('relu')
plt.grid()
plt.show()


def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

## W - Weights; b - Biases
def train(X, y, n_hidden, learning_rate, n_iter):
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1,1))
    for i in range(1, n_iter+1):
        Z2 = np.matmul(X, W1) + b1
        A2 = sigmoid(Z2)
        Z3 = np.matmul(A2, W2) + b2
        A3 = Z3
        ##
        dZ3 = A3 - y
        dW2 = np.matmul(A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0, keepdims=True)
        dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
        dW1 = np.matmul(X.T, dZ2)
        db1 = np.sum(dZ2, axis=0)
        ##
        W2 = W2 - learning_rate * dW2 / m
        b2 = b2 - learning_rate * db2 / m
        W1 = W1 - learning_rate * dW1 / m
        b1 = b1 - learning_rate * db1 / m
        ##
        if i % 100 == 0:
            cost = np.mean((y - A3) ** 2)
            print('Iteration %i, training loss: %f' %(i,cost))
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

##

## Trains a one-layer neural netowrk w/ 20 hidden units, a 0.1 learning rate, and 2000 iterations
from sklearn import datasets
housing = datasets.fetch_california_housing()
num_test = 10 #The last 10 samples as testing set
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X_train = housing.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = housing.data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = housing.target[-num_test:]

n_hidden = 20
learning_rate = 0.1
n_iter = 2000

model = train(X_train, y_train, n_hidden, learning_rate, n_iter)

## 

def predict(x, model):
    W1 = model['W1']
    b1 = model['b1']
    W2  = model['W2']
    b2 = model['b2']
    A2 = sigmoid(np.matmul(x, W1) + b1)
    A3 = np.matmul(A2, W2) + b2
    return A3

# Apply trained model on testing set
predictions = predict(X_test, model)

print(predictions[:, 0])

print(y_test)
