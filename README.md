# tensorflow_keras_samples
Started working with tensorflow and found an easier API to use keras.
Examples 1 and 2 use Tensorflow 
1 is a linear regression model but I think the y_data needs to be normalized in order to work correctly
2 is a fully connected neural network same issue with 1

Examples 3 and 4 use keras with tensorflow backend
3 is a linear regression
I normalized the y_data which fixed the issue I was having with example1
I think the large values of the y_data were giving the gradient descent on the loss function an issue because it was using mse
