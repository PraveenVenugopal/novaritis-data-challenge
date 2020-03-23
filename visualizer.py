import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter

sandbox = pd.read_csv("data/easy_load.csv").drop("Unnamed: 0",axis=1)
sandbox["Timestamp"]  = pd.to_datetime(sandbox["Timestamp"],format="%Y-%m-%d %H:%M:%S")
sandbox.set_index(["Timestamp"],inplace=True)
sandbox.sort_index()


# Plot the timestamps in original data in sequention order
# This gives insight into how many data points were not recorded

def plot_missing_times(table, start, end):
	st = start
	ll = []
	while st < end:
		if st in table:
			ll.append(1)
		else:
			ll.append(0)
		st += timedelta(minutes=10)
	plt.figure(figsize=(20,3))
	plt.title("Timestamp plot over given data")
	plt.scatter(np.arange(len(ll)), ll,
	            color=["red" if k == 0 else "blue" for k in ll])
	plt.show()
	return ll


st = sandbox.index[0]
en = sandbox.index[-1]
plot_missing_times(sandbox.index, st, en)



# Plot anomalies present in the given data
def plot_failures(sandbox):
	plt.figure(figsize=(20,5))
	plt.scatter(sandbox.index,[1 for _ in range(len(sandbox))],color="yellow")
	fail_piece = sandbox[sandbox.labels > 0]
	plt.scatter(fail_piece.index,[1 for _ in range(len(fail_piece))],color="red")
	plt.title("Anomaly distributions over time")
	plt.xlabel("Timestamp")
	plt.ylabel("Just a marker")
	plt.savefig("Timeline of Failure Occurrence.png")
	plt.show()

print("Plot anomalies in Complete data")
plot_failures(sandbox)

# TODO: Avoid hard coding of the "marked_index"
marked_index = datetime.strptime("2016-07-01 00:00:00","%Y-%m-%d %H:%M:%S")
train = sandbox.loc[:marked_index]
test = sandbox.loc[marked_index:]

print("Plot anomalies in Test data")
plot_failures(test)