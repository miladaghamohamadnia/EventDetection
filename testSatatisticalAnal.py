import pandas as pd
import numpy as np
import datetime as datetime
from matplotlib import pyplot as plt


path = '//Users//miladaghamohamadnia//Google Drive//Not categorized stuff//temp files//21aug2016_22PM.xlsx'
df = pd.read_excel(path)
df.sort_values(['Miliseconds'], ascending=[True], inplace=True)
df.reset_index(drop=True, inplace=True)
df = df[30000:]
timesSample = pd.to_datetime(df['Timestamps'])
totalseconds = (timesSample.max()-timesSample.min()).total_seconds()
samples = np.array(df.Values)


def autocorr(x, t):
    # print np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))
	n = len(x)
	return np.corrcoef(np.array([x[:n-t], x[t:]]),ddof=0)[0,1]

def fun(x):
    n = len(x)
    x = x-x.mean()
    # determine best fit line
    par = np.polyfit(range(len(x[x>0])), x[x>0], 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]

    return n*(n+2)*container



ts_log = df.Values[:-5]
ts_log = ( ts_log - ts_log.shift() )
plt.subplot(311)
plt.plot(np.abs(ts_log))

plt.subplot(312)
plt.plot((ts_log), color='red')


plt.subplot(313)
plt.plot(ts_log.rolling(3).median(), color='black')

plt.show()

