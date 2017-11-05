import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.io import loadmat  
from sklearn.preprocessing import OneHotEncoder  

# sigmoid function
def sigmoid(a):
    return 1/(1+np.exp(-a))

def sigmoid_gradient(z):  
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

#forward propogation function
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values = np.ones(m), axis = 1)

    z2 = a1 * theta1.T
    
    a2 = np.insert(sigmoid(z2), 0, values = np.ones(m), axis = 1)
    
    z3 = a2 * theta2.T
    
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h
    
def cost(params, input_size, hidden_size, num_labels, X, y, learn_rate):
    m = X.shape[0]
    
    X = np.matrix(X)
    y = np.matrix(y)
    
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))  

    return J

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):  
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####

    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

datasetPD = pd.read_csv("BitcoinOnly.csv", 
                        parse_dates=[0],
                        names = ("Date", "Open", "High", "Low", "Close", "Volume", "Market", "Variance", "Volatility"),
                        usecols=[0,1,2,3,4,5,6,9,10], 
                        index_col=0,
                        skiprows=1)

inputFeatures  = pd.DataFrame(datasetPD, columns = ["High", "Low","Open", "Close", "Variance", "Volume", "Market", "Volatility"])
data = loadmat('ex3data1.mat')

X = pd.DataFrame(datasetPD, columns = ["High", "Low","Open", "Close", "Variance", "Volume", "Volatility"])
y = pd.DataFrame(datasetPD, columns=["Market"])

X_T = data['X']
y_T = data['y']

#turns y data from 5000x1 array to a 5000x10 array of vectors
encoder = OneHotEncoder(sparse=False)
y_T_onehot = encoder.fit_transform(y_T)

# initial setup
input_size_dat = 8  
hidden_size_dat = 11  
num_labels_dat = 1 
learning_rate_dat = 1

input_size = 400  
hidden_size = 25  
num_labels = 10 
learning_rate = 1

layer1_weights_array = np.random.normal(0, 1, [input_size, hidden_size]) 
layer1_biases_array = np.zeros((1, hidden_size))

layer2_weights_array = np.random.normal(0, 1, [hidden_size, num_labels]) 
layer2_biases_array = np.zeros((1, num_labels))

params = (np.random.random
          (size=hidden_size * 
           (input_size + 1) + 
           num_labels * (hidden_size + 1)) 
          - 0.5) * 0.25
          
m = X_T.shape[0]  
X_T = np.matrix(X_T)  
y_T = np.matrix(y_T)

# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

print(theta1.shape)
print(theta2.shape)

a1, z2, a2, z3, h = forward_propogate(X_T, theta1, theta2)
print(a1.shape) 
print(z2.shape) 
print(a2.shape) 
print(z3.shape) 
print(h.shape)

print(cost(params, input_size, hidden_size, num_labels, X_T, y_T_onehot, learning_rate))

  




    