# tensorflow_keras_samples
Started working with tensorflow and found an easier API to use keras.

Examples 1 and 2 use Tensorflow directly

1 is a linear regression model but all the data needs to be normalized to work correctly

2 is a fully connected neural network same issue with 1

The tensorflow api is not that intuitive and, especially with the high level api, you don't have any idea what is running under the hood (ie. optimizer, loss function, learning rate, activation functions, ititial weight distributions) which is why I moved on to use keras.

Examples 3 and 4 use keras with tensorflow backend

3 is a linear regression.
I normalized all the data including the labels which fixed the issue I was having with example1.
I think the large values of the labels were giving the gradient descent instability issues. Especially using mse as the loss function

4 is a fully connected neural network.
Again I normalized all the data. The predicted output just has to be rescaled using the same parameters.
Uses a different optimization algorithm called 'adam' which simliar to adaDelta
