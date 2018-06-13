from __future__ import absolute_import, division, print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#linear regressor
def main():
	#loading data
	df = pd.read_csv("california_housing_train.csv")
	df = df.reindex(np.random.permutation(df.index))

	#removing data that seemed to be capped at 500000
	df = df[df["median_house_value"] < 250000]
	#drops unused features
	df.drop(df.columns[[0,1,2,3,4,5,6]], axis = 1, inplace = True)

	#normalize but preserve our y data value
	# y_data = df['median_house_value']
	df_norm = ((df - df.min()) / (df.max()-df.min()))
	# df_norm['median_house_value'] = y_data

	print(df_norm.describe())

	# graph = df.plot(x="median_income", y="median_house_value", style='o', markersize=1)
	# plt.show()

	#train and test features
	x_train = df_norm.sample(frac=0.9)
	x_test = df_norm.drop(x_train.index)

	#train and test labels
	y_train = x_train.pop("median_house_value")
	y_test = x_test.pop("median_house_value")

	#creating model
	model = Sequential()
	model.add(Dense(1, input_shape=(1,), activation='linear', init='uniform'))
	model.summary()

	#callBack
	callBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

	#stochastic gradient descent optimizer
	#learn rate 0.7 for (linear regression, sgd, mse) 7000 for (linear regression, sgd, logcosh)
	sgd = keras.optimizers.SGD(lr=0.01)

	#using logcosh loss function
	#similar to mean squared error
	model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=5, batch_size=1,  verbose=1)

	score1 = model.evaluate(x_test, y_test, verbose=0)
	score2 = model.evaluate(x_train, y_train, verbose=0)

	predictions = model.predict(x_test)

	# for i, pred in enumerate(predictions):
	# 	print("**********")
	# 	print(y_test.values[i])
	# 	print(pred)

	#graph of actual values in green predicted value in red
	plt.scatter(x_test, y_test, edgecolors='g')
	plt.scatter(x_test, predictions, edgecolors='r')
	plt.show()

	# print(score1[0] ** .5)
	# print(score2[0] ** .5)
	print(score1[0])
	print(score2[0])


if __name__ == "__main__":
	main()