import numpy as np
import pandas as pd
import os
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.models import load_model
from keras.losses import mean_squared_error, mean_absolute_error, logcosh, \
	categorical_crossentropy, sparse_categorical_crossentropy
from keras.callbacks import TerminateOnNaN, LearningRateScheduler, EarlyStopping
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(2)
seed(1)

data_path = "data/"
model_path = "models/"
image_path = "img/"



def construct_forecast_ensemble():
	if not os.path.exists(data_path):
		print("DATA PROCESSING NOT DONE : run data_processor.py first")
		return
	if not os.path.exists(model_path):
		os.mkdir(model_path)
	if not os.path.exists(image_path):
		os.mkdir(image_path)

	print("READING DATA -------------------------> ")
	train_data = pd.read_csv(data_path+"train_data.csv")
	test_data = pd.read_csv(data_path+"test_data.csv")
	train_labels = pd.read_csv(data_path+"train_labels.csv",header=None).values.reshape(-1)
	test_labels = pd.read_csv(data_path+"test_labels.csv",header=None).values.reshape(-1)

	# Split columns. We only need numeric columns for forecast
	cat_cols = [k for k in list(train_data.columns) if train_data[k].dtype != "float64"]
	num_cols = list(set(list(train_data.columns))-set(cat_cols))


	print("SETTING UP FORECAST PARAMETERS --------------------> ")
	# Denotes the number of features. ( Each feature is a univariate function w.r.t time)
	n_features = 1

	# Denotes the number of time steps we wish to process at once
	# Each entry in our data measured to 10 minutes => (60 minutes = 1hr)
	n_steps = 6


	# make univariate 1-D sequence to processible 2-D array of shape (n_steps,n_features)
	def process_sequence(seq,time_steps):
		data = []
		value = []
		i = time_steps
		while i < len(seq):
			data.append(seq[i-time_steps:i])
			value.append(seq[i])
			i+=1
		return np.asarray(data),np.asarray(value)


	# Callback parameters to stop training under certain conditions
	cb1 = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=0,
	                    mode='auto', restore_best_weights=False)
	cb2 = TerminateOnNaN()
	cb = [cb1, cb2]


	prediction_store = {}
	def columnwise_forecast(col):
		X, y = process_sequence(train_data[col], n_steps)
		n_samples = len(X)
		X = X.reshape(n_samples, n_steps, n_features)
		print("Data processed to feed shape ",X.shape, y.shape)

		# LSTM Model
		model = Sequential()
		model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
		model.add(Dense(1))
		model.compile(optimizer="adam", loss=logcosh)

		model.fit(X, y, epochs=1, verbose=1, callbacks=cb)
		tst_X, tst_y = process_sequence(test_data[col], n_steps)
		tst_X = tst_X.reshape(tst_X.shape[0], n_steps, n_features)

		yhat = model.predict(tst_X, verbose=0)
		print("Saving model for ",col)
		model.save(model_path + col.replace(".", "_") + "_predictor_model.h5")
		del model

		prediction_store[col] = (tst_y, yhat)

	print("STARTING MODEL CONSTRUCTION AND TRAINING -------------------> ")
	for col in num_cols[:2]:
		columnwise_forecast(col)

	def plot_preds_new(tst_y, yhat, col=""):
		fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 15),gridspec_kw={'height_ratios': [1, 5]})
		ax[0].scatter(np.arange(len(test_labels)), [1 for _ in range(len(test_labels))], color="yellow")
		fail_piece = np.where(test_labels[n_steps:] > 0)[0]
		ax[0].scatter(fail_piece, [1 for _ in range(len(fail_piece))], color="red")
		ax[0].set_title("Anomaly distributions over time")

		ax[1].plot(np.arange(len(tst_y)), tst_y, "r--")
		ax[1].plot(np.arange(len(yhat)), yhat, "b--")
		ax[1].set_title("Anomaly distributions over" + str(col))
		ax[1].set_xlabel("Timestamp")
		fig.savefig(image_path+str(col) + "ts_6.png")
		fig.show()

	for k in prediction_store:
		plot_preds_new(prediction_store[k][0],prediction_store[k][1],k)



construct_forecast_ensemble()