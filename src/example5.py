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

	df = df[df["median_house_value"] < 490000]
	df.drop(df.columns[[]], axis = 1, inplace = True)

	#normalize but preserve our y data value
	y_data = df['median_house_value']
	max_value = df['median_house_value'].max()
	min_value = df['median_house_value'].min()
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
	model.add(Dense(5, input_shape=(8,), activation='tanh', init='uniform'))
	model.add(Dense(1, activation='linear', init='uniform'))
	model.summary()

	#callBack - gives view of internal states
	callBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

	#stochastic gradient descent optimizer
	adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	#using logcosh loss function
	#similar to mean squared error
	model.compile(loss='logcosh', optimizer=adam, metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=2000, batch_size=1000,  verbose=2)

	score1 = model.evaluate(x_test, y_test, verbose=0)
	score2 = model.evaluate(x_train, y_train, verbose=0)

	predictions = model.predict(x_test)

	print(convert_value(score1[0], max_value, min_value))
	print(convert_value(score2[0], max_value, min_value))
	new_y = convert_value(y_test, max_value, min_value)
	new_pred = convert_value(predictions, max_value, min_value)
	plt.scatter(x_test['median_income'], new_y, edgecolors='g')
	plt.scatter(x_test['median_income'], new_pred, edgecolors='r')

	plt.show()
	average_error = 0
	count = 0
	for i, pred in enumerate(new_pred):
		# print("**********")
		# print(new_y.values[i])
		# print(pred[0])
		count += 1
		average_error += abs(new_y.values[i]-pred[0])
	average_error /= count
	print(average_error)


def convert_value(value, max_value, min_value):
	return value*(max_value-min_value) + min_value

if __name__ == "__main__":
	main()