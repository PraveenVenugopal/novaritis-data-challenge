import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from datetime import datetime, timedelta

data_path = "data"

if not os.path.exists(data_path):
    os.mkdir(data_path)

print("DATA READING -------------------------> ")
print("Start reading data")
xl_data  = pd.read_excel("IE01.552.051-data_pack.xlsx",sheet_name="Sensor_Data")
failure = pd.read_excel("IE01.552.051-data_pack.xlsx",sheet_name="Failure_Data")
print("Reading Data complete ",xl_data.shape,failure.shape)


print("\n\nDATA PROCESSING ------------------------------> ")
print("Process failure data")
# Standardize the start and end time of failures.
failure["FailureStartDate"] = failure["FailureStartDate"].apply(
    lambda x:datetime.strptime(x,"%d.%m.%Y").date())
failure["FailureStartTime"] = failure["FailureStartTime"].apply(
    lambda x:datetime.strptime(x,"%H:%M:%S").time())

# Use standardized timestamps to set an interval
failure["Fstart"] = [datetime.combine(
    failure.iloc[k]["FailureStartDate"],failure.iloc[k]["FailureStartTime"])
    for k in range(len(failure))]
failure["Fend"] = [failure.iloc[k]["Fstart"]+timedelta(
    hours=failure.iloc[k]["actual work\nin hours"]) for k in range(len(failure))]


# Setting type of failure within frame
# 1 - Non-maintenance ; 2 - Deferred Maintenance ; 3 - Immediate Maintenance
failure["Type"] = failure["Order Type"].apply(lambda x:int(x[-1])+1)


# Construct a list of "Timestamps" when there was an active failure.
failure_intervals = pd.Series()
for k in range(len(failure)):
    s = failure.iloc[k]["Fstart"].replace(second=0)
    while s <= failure.iloc[k]["Fend"]:
        failure_intervals[s] = failure.iloc[k]["Type"]
        s+=timedelta(minutes=1)   #make it per minute to have higher granuarity

print("Completed processing failure data")

def failtype(ts):
    if ts in failure_intervals.index:
        return failure_intervals[ts]
    else:
        return 0

# Make a copy of data for processing and storing
sandbox = xl_data.copy()
sandbox["labels"] = sandbox["Timestamp"].apply(failtype)
sandbox.to_csv("data/easy_load.csv")
print("Constructed Labels using failure time stamps")


sandbox.set_index(["Timestamp"],drop=True,inplace=True)
sandbox.sort_index(inplace=True)

features = list(sandbox.columns)
features.remove("labels")
cat_cols = ["AI552051.754_ALM","BACT.552051","BCMPLT.552051","FAL552051.754",
            "HV552051.331","HV552051.332","LAH552051.670","LAH552051.678",
            "LAH552051.680","M552051.801","M552051.802","M552051.823","M552051.826",
            "M552051.871","MAINT.552051","MODMAN.552051","PARTREC.552051",
            "QCA552051_001","RUN.552051","SIC552051.801_ALM","SSOALM.552051",
            "VI552051.748_ALM","ZS552051.737","ZS552051.740","ZS552051_753"]
non_cat = list(set(features)-set(cat_cols))


print("OHE on categorical columns")
# OHE of categorical columns
for col in cat_cols:
    sandbox[col] = sandbox[col].astype("category")
    sandbox = pd.get_dummies(sandbox, prefix=[col], columns=[col])


print("Conversion on numeric columns")
# Remove Bad Input and I/O Timeout from Numeric columns
for col in non_cat:
    sandbox[col] = pd.to_numeric(sandbox[col],errors="coerce").fillna(sandbox[col])
    sandbox[col+"_Bad Input"] = [1 if k == "Bad Input" else 0
                                 for k in sandbox[col].tolist()]
    sandbox[col+"_I/O Timeout"] = [1 if k == "I/O Timeout" else 0
                                   for k in sandbox[col].tolist()]
sandbox[non_cat] = sandbox[non_cat].apply(pd.to_numeric, errors='coerce').fillna(0)


# TODO: Avoid hard coding of the "marked_index"
marked_index = datetime.strptime("2016-07-01 00:00:00","%Y-%m-%d %H:%M:%S")

# Get fresh set of feature columns since we added a bunch of those  during OHE
feat_cols = list(sandbox.columns)
feat_cols.remove("labels")

print("\n\nDATA SAVE ----------------------> ")
print("Split test/train and store to path : ",str(data_path))
train_data = sandbox.loc[:marked_index][feat_cols]
test_data = sandbox.loc[marked_index:][feat_cols]
train_labels = sandbox.loc[:marked_index]["labels"]
test_labels = sandbox.loc[marked_index:]["labels"]
print("Train data to be stored ",train_data.shape,train_labels.shape)
print("Test data to be stored ",test_data.shape,test_labels.shape)

train_data.to_csv("data/train_data.csv",index=False)
test_data.to_csv("data/test_data.csv",index=False)
train_labels.to_csv("data/train_labels.csv",index=False)
test_labels.to_csv("data/test_labels.csv",index=False)
print("DATA PROCESSING COMPLETE")
