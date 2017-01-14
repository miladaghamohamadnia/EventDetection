from firebase import firebase
from pandas import ExcelWriter
import pandas as pd
import numpy as np
import datetime
import time
import json
import random

def frbsSorter(REF):
	miliseconds = []
	timestamps  = []
	values      = []
	# keys        = []
	k = 0
	result = REF.get('/001', None)
	for key, value in result.items():
	    k+=1
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

def flag(ref, Id):
	lat = random.uniform(40.58, 40.70)
	lon = random.uniform(-73.88, -74.04)
	loc = random.uniform(0, 1)
	direction = 'N'
	data = { 'id':Id, 'lat':lat, 'lon':lon, 'loc':loc, 'direction':direction }
	data_string = json.dumps(data)
	ref.post('/flag', data_string)


# minuteslimit = 1
# rate = 40.0  # in miliseconds
# N = minuteslimit*60*1000/rate  # number of records during minuteslimit to be kept
ref = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)
# allCurrentRecords = frbsSorter(ref)
# preProcess(allCurrentRecords)


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


while True:
	ref.delete('flag', None)
	for i in range(10):
		flag(ref, 'ttttesttt')
		time.sleep(0.05)





