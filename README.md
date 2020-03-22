# Feedforward-Deep-Neural-Network
a vectorized, python-based implementation of feedforward deep neural network for binary classification. The neural network has been made from scratch.

-> Limitations:

1. For now, Only tanh, sigmoid, relu and leakyrelu activation functions are supported in the hidden layers.
2. Only sigmoid activation function is supported in the output layer (Binary Classification).


-> The neural network class has the following three major methods:

1. train(self, trainSet, trainLabels, noOfIterations=10, learningRate=0.01, validationSet=None, validationLabels=None ) : the train function is passed as parameters training dataset (trainSet), training dataset labels (trainLabels), noOfiterations, learningRate, validationSet and validationLabels. This funtion then learns weights.

2. addFeedForwardLayer(self, activationFunction='sigmoid', noOfUnits=10) : this function is used to add a hidden layer or the output layer in the model. All calls to this function must be made before the train function is called. You can add as many hidden layers as you want. The last layer added using the last function call will be considered the output layer. The last (output) layer must be a sigmoid layer with 1 unit. The activation functions supported for hidden layers are sigmoid, tanh, relu and leakyrelu.

3. predict(self, testSet) : predict function is passed as parameter the test dataset (testSet). It then predicts the labels of each item in the test dataset and returns the labels in an array.


-> Note:

1. The input shape for training set, validation set, and test set must be (nx, m) where nx is the number of features, and m is the number of items in the set.

2. The shape of arrays containing labels for training set, test set and validation set must be (1, m) where m is the number of items.

3. The model has been trained and tested in main.py on a dataset containg cat images (dataset has been taken from coursera deep learning course assignment). The model achieves the test accuracy of 76%.
