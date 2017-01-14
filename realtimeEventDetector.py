from firebase import firebase
from pandas import ExcelWriter
import pandas as pd
import numpy as np
import datetime
import time
import json
import random
import datetime as datetime
from matplotlib import pyplot as plt
# import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from matplotlib.dates import DateFormatter
import statsmodels.api as sm
from scipy.stats import invgauss, expon


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

# path = '//Users//miladaghamohamadnia//Google Drive//Not categorized stuff//temp files//21aug2016_22PM.xlsx'
# df = pd.read_excel(path)
REF = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)
df = frbsSorter(REF)
df.columns = ['Miliseconds', 'Timestamps', 'Values']

df.sort_values(['Miliseconds'], ascending=[True], inplace=True)
df.reset_index(drop=True, inplace=True)
df = df[30000:]
timesSample = pd.to_datetime(df['Timestamps'])
totalseconds = (timesSample.max()-timesSample.min()).total_seconds()
samples = np.array(df.Values)
freq = samples.shape[0]/totalseconds




def autocorr(sig, lag):
	n = len(sig)
	return np.corrcoef(np.array([sig[:n-lag], sig[lag:]]),ddof=0)[0,1]
def JenkinsFun(x):
    n = len(x)
    h = int(np.sqrt(np.log(n)))
    container = 0
    for lag in range(1,h+1):
        rov = autocorr(x, lag)**2/(n-lag)
        container =+ rov
    return n*(n+2)*container
def DiffFun(x):
    	mid = int(len(x)/2)
    	return x[mid]-x.mean()
def invGaus_fit(data):
	dist = invgauss
	dist_fit = dist.fit(data)		# fit a distribution to the data
	dist_model = dist(*dist_fit)
	dist_mu = dist_model.mean()		# find the distribution mean
	dist_std = dist_model.std()		# find the distribution standard deviation
	return dist_mu, dist_std
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='same')

def invGaus_outliers(pop, freq, invGaus_factor, cluster_dist, bw_max, smooth_N, width_exagg):
	print "###  invGaus_outliers  ###"
	dist_mu, dist_std = invGaus_fit(pop)
	print dist_mu, dist_std
	# find the inverse gaussian lower confidence interval
	CI_l = dist_mu-invGaus_factor*dist_std
	excessVals = pop[pop<CI_l].index.tolist()
	if len(excessVals)<1: return [],[]
	outliers = []
	temp = excessVals[0]
	container = []
	for el in excessVals:
		if (el - temp) < 2:
			container.append(el)
		else:
			outliers.append(pop[pop==pop[container].min()].index.tolist()[0])
			container=[]
			container.append(el)
		temp = el
	print "outliers:  ", outliers

	# cluster closer outliers
	temp = outliers[0]
	container = []
	for el in outliers:
		if (el - temp) > int(cluster_dist*freq):  	# Not close!
			container.append(el)
		else:
													# They're close!
			if pop[el] <= pop[temp]:
				if container:
					container.pop()
				container.append(el)
		temp = el
	outliers = container
	print "outliers after clustering:  ", outliers

	# find the width of events
	data = pop
	trshd = data.mean()
	# print trshd
	bandwidths = []
	for event_center in outliers:
		container = []
		k=0
		bw_l=event_center-1
		bw_h=event_center+1
		while k<int(freq*bw_max):
			if data[bw_l]<=trshd:
				bw_l-=1
			else: break
			k=k+1
		k=0
		while k<int(freq*bw_max):
			if data[bw_h]<=trshd:
				bw_h+=1
			else: break
			k=k+1
		bw = bw_h-bw_l
		bandwidths.append(bw/width_exagg)
		# print "BW: ", bw, bw/width_exagg
	return outliers, bandwidths

def Gaus_outliers(pop, freq, Gaus_factor, cluster_dist, bw_max, trshd_coeff, width_exagg, padding):
	print "###  Gaus_outliers  ###"
	pop = pop[padding+1:-padding-1]
	dist_mu = pop.mean()
	dist_std = pop.std()
	pop_abs = np.abs(pop)
	# find the inverse gaussian lower confidence interval
	CI_h = dist_mu+Gaus_factor*dist_std
	excessVals = pop_abs[pop_abs>CI_h].index.tolist()
	if len(excessVals)<1: return [],[]
	outliers = []
	temp = excessVals[0]
	container = []
	for el in excessVals:
		if (el - temp) < 2:
			container.append(el)
		else:
			outliers.append(pop_abs[pop_abs==pop_abs[container].max()].index.tolist()[0])
			container=[]
			container.append(el)
		temp = el
	print "outliers:  ", outliers

	# cluster closer outliers
	temp = outliers[0]
	container = []
	for el in outliers:
		if (el - temp) > int(cluster_dist*freq):  	# Not close!
			container.append(el)
		else:
													# They're close!
			if pop[el] <= pop[temp]:
				if container:
					container.pop()
				container.append(el)
		temp = el
	outliers = container
	print "outliers after clustering:  ", outliers

	# find the width of events
	data = pop.rolling(60).std()
	trshd = data.std()/trshd_coeff
	# print trshd
	bandwidths = []
	for event_center in outliers:
		container = []
		k=0
		bw_l=event_center-1
		bw_h=event_center+1
		while k<int(freq*bw_max):
			if data[bw_l]>=trshd:
				bw_l-=1
			else: break
			k=k+1
		k=0
		while k<int(freq*bw_max):
			if data[bw_h]>=trshd:
				bw_h+=1
			else: break
			k=k+1
		bw = bw_h-bw_l
		bandwidths.append(bw/width_exagg)
		# print "BW: ", bw, bw/width_exagg
	return outliers, bandwidths

# def anomaly_tester(j_o, j_b,d_o, d_b):
# 	if len(j_o)>len(d_o):
# 		for i in range():
			


df['data_diff'] = df.Values.rolling(3).apply(DiffFun)
df['JenkinsTest'] = df.Values.rolling(150).apply(JenkinsFun)
df.dropna(axis=0, how='any', inplace=True)
data_diff = df.data_diff
padding = 10
modif_diff = data_diff*data_diff.rolling(padding).std()
JenkinsTest = df.JenkinsTest
dist_mu = JenkinsTest.mean()	
dist_std = JenkinsTest.std()

jenkins_outliers, jenkins_bandwidths = invGaus_outliers(JenkinsTest,freq,invGaus_factor=3, cluster_dist = 20, bw_max=30, smooth_N=3, width_exagg=40)
print jenkins_outliers, jenkins_bandwidths
modif_diff_outliers, modif_diff_bandwidths = Gaus_outliers(modif_diff,freq,Gaus_factor=7, cluster_dist = 20, bw_max=100, trshd_coeff=5, width_exagg=15, padding=padding)
print modif_diff_outliers, modif_diff_bandwidths
# anomaly_tester(jenkins_outliers, jenkins_bandwidths,modif_diff_outliers, modif_diff_bandwidths)


##
if (len(jenkins_outliers)!=0 || len(jenkins_outliers)!=0):
	# Two subplots, unpack the axes array immediately
	f, (ax1, ax2, ax3, ax4, ax5,ax6) = plt.subplots(6, 1, sharex=True)
	ax1.plot(np.array(df.Timestamps)[:], JenkinsTest)
	ax1.axhline(dist_mu, color='k', linestyle='solid', lw = 2)
	ax1.axhline(dist_mu-1*dist_std, color='r', linestyle='solid')
	ax1.axhline(dist_mu-2*dist_std, color='r', linestyle='solid')
	ax1.axhline(dist_mu-3*dist_std, color='r', linestyle='solid')
	ax2.plot(np.array(df.Timestamps)[:], df.Values,'r')
	formatter = DateFormatter('%H:%M:%S')
	ax2.xaxis.set_major_formatter(formatter)
	labels = ax2.get_xticklabels()
	plt.setp(labels, rotation=30, fontsize=10)
	ax3.plot(np.array(df.Timestamps)[:], data_diff)
	modif_diff = data_diff*data_diff.rolling(10).std()
	ax4.plot(np.array(df.Timestamps)[:], modif_diff, 'red')
	data = (modif_diff[11:-11])
	ax5.plot(np.array(df.Timestamps)[11:-11], data)
	ax5.axhline(data.mean(), color='k', linestyle='solid', lw = 2)
	ax5.axhline(data.mean()-5*data.std(), color='r', linestyle='solid')
	ax5.axhline(data.mean()+5*data.std(), color='r', linestyle='solid')
	ax6.plot(np.array(df.Timestamps)[11:-11],data.rolling(20).std())
	data = modif_diff[11:-11]
	# plt.figure()
	# bins = np.linspace(data.min(), data.max(), 1000)
	# plt.hist(-np.abs(np.array(data)), bins)
	plt.show()

