


#!/usr/bin/env python
#coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
import pandas as pd
import datetime as datetime
from matplotlib import pyplot as plt
# import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.6, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    print cols
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
def plotstft(filepath, binsize=2**8, plotpath=None, colormap="jet"):
    df = pd.read_excel(path)
    df = df[30000:]
    df.sort_values(['Miliseconds'], ascending=[True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    timesSample = pd.to_datetime(df['Timestamps'])
    totalseconds = (timesSample.max()-timesSample.min()).total_seconds()
    samples = np.array(df.Values)
    samplerate = round(samples.shape[0]/totalseconds)
    def fun(x):
        mid = int(len(x)/2)
        return(x[mid]-x.mean())
    samples_diff = df.Values.rolling(5).apply(fun)


    print '# samples',samples.shape[0]
    print '# seconds',totalseconds
    print '# rate',samplerate
    samples = np.array(samples_diff)
    s = stft(samples, binsize)
    print s.shape
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    
    timebins, freqbins = np.shape(ims)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")

    xlocs = np.float32(np.linspace(0, timebins-1, 20))
    samplesinSeconds = ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate
    helper = np.vectorize(lambda x: datetime.timedelta(seconds=round(x)))
    dt_sec = timesSample.min() + helper(samplesinSeconds)

    plt.xticks(rotation=70)
    plt.xticks(xlocs, [l.strftime("%H:%M:%S") for l in dt_sec])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    ylim_ = axes.get_ylim()
    xlim_ = axes.get_xlim()
    print [ylim_[0]+0.1*(ylim_[1]-ylim_[0]),ylim_[1]]
    print [xlim_[0]+0.1*(xlim_[1]-xlim_[0]),xlim_[1]]
    axes.set_ylim([ylim_[0]+0.05*(ylim_[1]-ylim_[0]),ylim_[1]])
    axes.set_xlim([xlim_[0]+0.0*(xlim_[1]-xlim_[0]),xlim_[1]])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()
    
    plt.figure(figsize=(15, 7.5))
    plt.plot(ims.median(1))
    # plt.plot(samples_diff,'r')
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.show()


    




path = '//Users//miladaghamohamadnia//Google Drive//Not categorized stuff//temp files//21aug2016_22PM.xlsx'
plotstft(path)

