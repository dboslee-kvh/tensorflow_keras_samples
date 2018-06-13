from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import pandas as pd

def main(argv):
	dataframe = pd.read_csv("california_housing_train.csv")
	dataframe = dataframe[dataframe["median_house_value"] < 250000]

	y_data = dataframe['median_house_value']
	df_norm = ((dataframe - dataframe.min()) / (dataframe.max()-dataframe.min()))

	#useful info about data
	df_norm['median_house_value'] = y_data

	#train and test features
	x_train = df_norm.sample(frac=0.8)
	x_test = df_norm.drop(x_train.index)

	#train and test labels
	y_train = x_train.pop("median_house_value")
	y_test = x_test.pop("median_house_value")

	#train and test functions
	train_input_fn = lambda:input_fn(100, x_train, y_train, True, 1000)
	test_input_fn = lambda:input_fn(100, x_test, y_test)

	feature_columns = []

	# adds all feature columns to array 
	# ignoring the label we want to predict
	for item in dataframe:
		if item == "median_house_value":#or item == "longitude" or item == "latitude":
			continue
		feature_columns.append(tf.feature_column.numeric_column(key=item))
	# feature_columns = [
	# 	# "curb-weight" and "highway-mpg" are numeric columns.
	# 	tf.feature_column.numeric_column(key="median_income"),
	# ]

	model = tf.estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)

	model.train(input_fn=train_input_fn, steps=5000)

	eval_result = model.evaluate(input_fn=test_input_fn)
	average_loss = eval_result["average_loss"]

	print("\n" + 80 * "*")
	print("\nRMS error: ${:.0f}".format(average_loss**0.5))

	# predict_results = model.predict(input_fn=test_input_fn)

	# print(predict_results)

	# print("\nPrediction results:")
	# for i, prediction in enumerate(predict_results):
	# 	msg = ("median_house_value: {: 4f}, "
	# 		"Prediction: {: 9.2f}")
	# 	msg = msg.format(prediction["predictions"][0])
	# 	print("\n" + msg)
	# print()

def input_fn(batch, x, y=None, shuffle=False, shuffle_buffer_size=1000):
	if y is not None:
		dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
	else:
		dataset = tf.data.Dataset.from_tensor_slices(dict(x))
	if shuffle:
		dataset = dataset.shuffle(shuffle_buffer_size).batch(batch).repeat()
	else:
		dataset = dataset.batch(batch)

	return dataset.make_one_shot_iterator().get_next()

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main=main)