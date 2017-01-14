from firebase import firebase
from pandas import ExcelWriter
import pandas as pd
import numpy as np
import datetime
import time
import json
import random
import matplotlib.pyplot as plt

def frbsSorter(REF):
    miliseconds = []
    timestamps  = []
    values      = []
    # keys        = []
    k = 0
    result = REF.get('/002', None)
    for key, value in result.items():
        # print key,value
        k+=1
        if k>5000: 
            break
        ms = value['timestamp']
        timeS = datetime.datetime.fromtimestamp(ms/1000.0)
        timestamps.append(timeS)
        miliseconds.append(value['timestamp'])
        values.append(value['value'])
        # keys.append(key)
    data = np.vstack((np.array(miliseconds),np.array(timestamps))).T
    values = np.array(values)[:,np.newaxis]
    data = np.hstack((data,values))
    # keys = np.array(keys)[:,np.newaxis]
    # data = np.hstack((keys,data))
    df = pd.DataFrame(data=data, columns=['miliseconds','timestamps','value'])
    df.sort_values(['miliseconds'], ascending=[False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print df.head()
    print df.shape
    return df

def preProcess(df):
	print df.timestamps.max()-df.timestamps.min()
	# df['diff'].apply(lambda x: x + 1)


# minuteslimit = 1
# rate = 40.0  # in miliseconds
# N = minuteslimit*60*1000/rate  # number of records during minuteslimit to be kept
ref = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)
allCurrentRecords = frbsSorter(ref)
preProcess(allCurrentRecords)


# ref.delete('flag', None)
# flag(ref, '12')
# flag(ref, '46')



# while True:
	# minuteslimit = 1
	# rate = 40.0  # in miliseconds
	# N = minuteslimit*60*1000/rate  # number of records during minuteslimit to be kept
	# firebase = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)
	# clean_once(firebase, N)

	# ref.delete('flag', None)
	# for i in range(2):
	# 	flag(ref)
	# time.sleep(0.01*1)





