#The dataset is taken from the coursera assignment 1 of deep learning specialization course 1

import h5py

import numpy as np

def load_dataset():
    train_dataset = h5py.File('./dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('./dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



import numpy as np

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


#flattening each image in both training set and test set
print(train_set_x_orig.shape)
print(test_set_x_orig.shape)

train_set_x=train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x=test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

#normalize both training set and test set
train_set_x=train_set_x/255
test_set_x=test_set_x/255

train_set_x=train_set_x.T
test_set_x=test_set_x.T


from  model import FeedForwardNeuralNetwork
np.random.seed(100)

nn= FeedForwardNeuralNetwork()

#adding feed forward layers
nn.addFeedForwardLayer(activationFunction="leakyrelu", noOfUnits=5) #1st layer

nn.addFeedForwardLayer(activationFunction="sigmoid",noOfUnits=1) #last layer (the last layer must be sigmoid layer with a single unit)

nn.train(train_set_x, train_set_y, noOfIterations=1600, learningRate=0.005)
predictions=nn.predict(test_set_x)

#print test accuracy
print("Test Accuracy: {}%".format(100 - np.mean(np.abs(predictions - test_set_y)) * 100))
