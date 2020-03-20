# Feed-Forward-Neural-Network
a vectorized, python-based implementation of deep feed forward neural network for binary classification.

-> Limitations:

1. For now, Only tanh, sigmoid, relu and leakyrelu activation functions are supported in the hidden layers.
2. Only sigmoid activation function is supported in the output layer (Binary Classification).


-> The following functions are supported:

train(self, trainSet, trainLabels, noOfIterations=10, learningRate=0.01, validationSet=None, validationLabels=None ) : the train function is passed as parameters training dataset (trainSet), training dataset labels (trainLabels), noOfiterations, learningRate, validationSet and validationLabels. This funtion then learns weights.

addFeedForwardLayer(self,activationFunction='sigmoid', noOfUnits=10): this function is used to add a new layer in the model. All calls to this function must be made before the train function is called. The last layer must be a sigmoid layer with 1 unit. The activation functions supported for hidden layers are sigmoid and tanh.

predict(self, testSet): predict function is passed as parameter the test dataset (testSet). It then predicts the labels of each item in the test dataset and returns the labels in an array.


-> Note:

1. The input shape for training set, validation set, and test set must be (nx, m) where nx is the number of features, and m is the number of items in the set.

2. The shape of arrays containing labels for training set, test set and validation set must be (1, m) where m is the number of items.

3. The model has been trained and tested in main.py on a dataset containg cat images (dataset has been taken from coursera deep learning course assignment). The model achieves test accuracy of 74%.
