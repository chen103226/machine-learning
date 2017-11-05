import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset
from lr_utils import load_dataset

#loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# Example of a picture
index = 22
plt.imshow(train_set_x_orig[index])
#print(train_set_x_orig[index])
#print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

m_train = train_set_x_orig.shape[0]
m_test =test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape( train_set_x_orig.shape[1]* train_set_x_orig.shape[2]*3,train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x_orig.reshape( test_set_x_orig.shape[1]* test_set_x_orig.shape[2]*3,test_set_x_orig.shape[0])

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
print(train_set_x)

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return z

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def propagate(w,b,X,Y):
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -1/m*np.sum(Y)