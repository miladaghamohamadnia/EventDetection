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

""" short time fourier transform of sample signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

""" plot spectrogram"""
def plotstft(REF, binsize=2**7, plotpath=None, colormap="jet"):
    df = frbsSorter(REF)
    print df.head()
    timesSample = pd.to_datetime(df['timestamps'])
    totalseconds = (df.miliseconds.max()-df.miliseconds.min())/1000
    samples = np.array(df.value)
    samplerate = round(samples.shape[0]/totalseconds)
    
    print '# samples',samples.shape[0]
    print '# seconds',totalseconds
    print '# rate',samplerate
    s = stft(samples, binsize)
    
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    ims = np.transpose(ims)
    timebins, freqbins = np.shape(ims)
    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(ims[:,:], origin="lower", aspect="auto", cmap=colormap, interpolation="nearest")
    ax2.plot(np.array(df.timestamps)[::-1], np.array(df.value))
    plt.show()
    plt.figure(figsize=(15, 7.5))
    plt.plot(np.transpose(ims)[0,:])
    plt.show()


    from mpl_toolkits.mplot3d import Axes3D
    ims = ims[2:,:]
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:ims.shape[0], 0:ims.shape[1]]
    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, ims ,rstride=1, cstride=1, cmap=plt.cm.jet,
            linewidth=0)
    # show it
    plt.show()





def frbsSorter(REF):
    	miliseconds = []
	timestamps  = []
	values      = []
	# keys        = []
	k = 0
	result = REF.get('/002', None)
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
    




ref = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)
plotstft(ref)
plt.close(plt.gcf())

