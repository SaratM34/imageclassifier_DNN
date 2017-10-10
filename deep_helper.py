import numpy as np 
import time
import h5py
import scipy
from PIL import Image 
from scipy import ndimage
from dnn_app_utils_v2 import *
import matplotlib.pyplot as plt 
from testCases_v3 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
import pickle
#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

#np.random.seed(1)

# 2-Layer Network

# Initialize parameters

def initialize_parameters(n_x, n_h, n_y):

	np.random.seed(1)
	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))

	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}

	return parameters

# Initializing parameters for Deep L-layer Network

def initialize_parameters_deep(layer_dims):

	np.random.seed(1)
	parameters = {}
	L = len(layer_dims)
	for l in range(1,L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

	return parameters

# Forward propagation

def linear_forward(A, W, b):

	Z = np.dot(W,A) + b # We are passing 'W' is a row vector
	cache = (A, W, b) # tuple

	return Z, cache

# Combined linear and activation functions for 2-layer NN

def linear_activation_forward(A_prev, W, b, activation):

	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)

	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)

	cache = (linear_cache, activation_cache)

	return A, cache

# Testing
'''A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))'''

def L_model_forward(X, parameters):

	caches = []
	A = X
	L = len(parameters)//2

	for l in range(1,L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
	caches.append(cache)

	return AL, caches

# compute cost

def compute_cost(AL, Y):
	m = Y.shape[1]

	cost = (-1/m) * (np.dot(np.log(AL),Y.T) + np.dot(np.log(1-AL),(1-Y).T))
	cost = np.squeeze(cost)
	return cost

'''Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))'''

#Back Propagation

# Computing derivative of loss function w.r.t linear function(Z)

def linear_backward(dZ, cache):



	A_prev, W, b = cache
	m = A_prev.shape[1] 

	dW = (1/m) * np.dot(dZ, A_prev.T)
	db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
	dA_prev = np.dot(W.T,dZ)

	return dA_prev, dW, db

#Linear Activation backward

def linear_activation_backward(dA, cache, activation):

	linear_cache, activation_cache = cache
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	if activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

#Backprop for L-layer's

def L_model_backward(AL, Y, caches):

	grads = {}
	L = len(caches)
	m = AL.shape[1]

	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	current_cache = caches[-1]
	grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")



	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)], current_cache, activation="relu")

		grads["dA"+str(l+1)] = dA_prev_temp
		grads["dW"+str(l+1)] = dW_temp
		grads["db"+str(l+1)] = db_temp

	return grads


'''AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)'''


#Update parameters

def update_parameters(parameters, grads, learning_rate):

    
    L = len(parameters) // 2 

    for l in range(1,L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate)*grads["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate)*grads["db"+str(l)]

    return parameters


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


m_train = train_x_orig.shape[0] 
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255. 
test_x = test_x_flatten/255.

# 2 layer model

'''n_x = 12288
n_h = 7
n_y = 1

layers_dims = (n_x, n_h, n_y)

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations= 3000, print_cost= False):

	np.random.seed(1)
	grads= {}
	costs = []
	m = X.shape[1]
	(n_x,n_h,n_y) = layers_dims
	parameters = initialize_parameters(n_x,n_h,n_y)

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(num_iterations):
		A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
		A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

		cost = compute_cost(A2, Y)

		dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

		dA1, dW2, db2 = linear_activation_backward(dA2, cache2,activation = "sigmoid")
		dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

		grads['dW1'] = dW1
		grads['db1'] = db1
		grads['dW2'] = dW2
		grads['db2'] = db2

		parameters = update_parameters(parameters, grads, learning_rate)

		W1 = parameters["W1"]
		b1 = parameters["b1"]
		W2 = parameters["W2"]
		b2 = parameters["b2"]

		if print_cost and i % 100 == 0:
			print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
		if print_cost and i % 100 == 0:
			costs.append(cost) 

	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
    
	return parameters

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)'''

# L-layer model

layers_dims = [12288, 20, 7, 5, 1]

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        
        AL, caches = L_model_forward(X, parameters)
       
        
        # Compute cost.
        
        cost = compute_cost(AL, Y)
        
    
        # Backward propagation.
        
        grads = L_model_backward(AL, Y, caches)
        
 
        # Update parameters.
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost=True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)



# Predict for given image

my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


























