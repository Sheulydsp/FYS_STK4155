import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns




def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#we now define the cost/loss function. We use the MSE.
def MSE(y_data,y_model):
    n = np.size(y_model)
    y_data = y_data.reshape(-1,1)
    y_model = y_model.reshape(-1,1)
    return np.sum((y_data-y_model)**2)/n

#We define the R square
def rsquare(y, ypredict):
    y = y.reshape(-1, 1)
    ypredict = ypredict.reshape(-1,1)
    return 1-(np.sum((y-ypredict)**2)/np.sum((y-np.mean(y))**2))


#define sigmoid as the activation function for the hidden layer
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#define linear function, we'll use it for the output since we have a continous output.
#It's here just for showing how it's defined, but since the linear funcion is just returning the value of the parameter, we won't need to use it later in the code.
def linear(x):
    return x


#We now write the feed forward pass
def feed_forward(X,h_w, h_b):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, h_w) + h_b
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    
    return z_o

#feed forward stops before the output
def feed_forw_hid(X,h_w, h_b):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, h_w) + h_b
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    return a_h


#We define the backpropagation algorithm
def backpropagation(X,Y):
    a_h = feed_forw_hid(X)
    #output error
    output_error = feed_forward(X) - Y
    #hidden error
    hidden_error = np.matmul(output_error, output_weights.T) * a_h * (1 - a_h)

    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, output_error)
    output_bias_gradient = np.sum(output_error, axis=0)
    
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, hidden_error)
    hidden_bias_gradient = np.sum(hidden_error, axis=0)
    #Updated weights and bias are returned
    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient


#We define the learning of the NN.
def learn(ld, et, o_w, o_b, h_w, h_b):
        num_epochs = 1000
        for epoch in range(num_epochs):
                # calculate gradients
                dWo, dBo, dWh, dBh = backpropagation(X_train, z_train)    
                #regularization term gradients
                dWo += ld * output_weights
                dWh += ld * h_w

                #update weights and biases
                o_w -= et * dWo
                o_b -= et * dBo
                h_w -= et * dWh
                h_b -= et * dBh
        #we find the loss by calculating the mse with the predicted value and the actual value.
        #We find mse for train and test.
        ls = MSE(z_train, feed_forward(X_train,h_w, h_b))
        r = rsquare(z_train, feed_forward(X_train,h_w, h_b))
        ls_test = MSE(z_test, feed_forward(X_test,h_w, h_b))
        r_test = rsquare(z_test, feed_forward(X_test, h_w, h_b))
        return ls, r, ls_test, r_test