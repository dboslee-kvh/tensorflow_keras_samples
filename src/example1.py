from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.data import Dataset
import matplotlib.pyplot as plt

def main(argv = None):
	#load data from csv
	dataframe = pd.read_csv("california_housing_train.csv")
	print(dataframe.keys())
	print(dataframe)

	# feature_name = "longitude"
	# feature_name = "latitude"
	# feature_name = "housing_median_age"
	# feature_name = "total_rooms"
	# feature_name = "total_bedrooms"
	# feature_name = "population"
	# feature_name = "households"
	feature_name = "median_income"

	#randomize order
	dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
	dataframe = dataframe[dataframe["median_house_value"] < 250000]
	# graph = dataframe.plot(x=feature_name, y="median_house_value", style='o', markersize=1)
	# plt.show()
	# df_norm = dataframe
	df_norm = (dataframe -dataframe.min()) / (dataframe.max()-dataframe.min())
	#useful info about data
	print(dataframe.describe())

	# train and test features
	x_train = df_norm.sample(frac=0.9)
	x_test = df_norm.drop(x_train.index)

	#train and test labels
	y_train = x_train.pop("median_house_value")
	y_test = x_test.pop("median_house_value")

	#train and test functions
	train_input_fn = lambda:input_fn(100, x_train, y_train, True, 1000)
	test_input_fn = lambda:input_fn(1000, x_test, y_test)

	feature_columns = []

	# adds all feature columns to array 
	# ignoring the label we want to predict
	# for item in dataframe:
	# 	if item == "median_house_value":
	# 		continue
	# 	feature_columns.append(tf.feature_column.numeric_column(key=item))
	feature_columns = [
		# tf.feature_column.numeric_column(key="households"),
		tf.feature_column.numeric_column(key=feature_name)
	]

	model1 = tf.estimator.LinearRegressor(feature_columns=feature_columns)
	model1.train(input_fn=train_input_fn, steps=2000)

	eval_results = model1.evaluate(input_fn=test_input_fn)
	# for key in sorted(eval_results):
	# 	print('%s: %s' % (key, eval_results[key]))

	average_loss = eval_results["average_loss"]

	print("\n" + 80 * "*")
	print(tf.train.get_global_step())
	print("\nRMS error: ${:.0f}".format(average_loss**0.5))

	predict_results = model1.predict(input_fn=test_input_fn)

	print(predict_results)

	predictions = []

	for i in predict_results:
		print(i['predictions'][0])
		predictions.append(i['predictions'][0])
	# print("\nPrediction results:")
	# for i, prediction in enumerate(predict_results):
	# 	msg = ("median_house_value: {: 4f}, "
	# 		"Prediction: {: 9.2f}")
	# 	msg = msg.format(prediction)
	# 	print("\n" + msg)
	# print()
	print(len(x_test))
	print(len(y_test))
	plt.scatter(x_test['median_income'], y_test, edgecolors='g')
	plt.scatter(x_test['median_income'], predictions, edgecolors='r')
	plt.show()

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