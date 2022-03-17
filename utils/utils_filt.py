from scipy.optimize import curve_fit
from scipy import signal as sig
import numpy as np

def filt(time, signal, zerosplus = 0, Type = 'BPF', Freqmin = 20, Freqmax=100):
    fs = abs(1/(time[-1]-time[-2]))

    argmax = signal.argmax()
    if argmax > len(time)/2:
        argmax=0
    time = time[argmax+zerosplus:]
    amp  = signal[argmax+zerosplus:]


    if Type == 'PolyExp':
        def f(t,a,b,c,d,e,t0):
            return(a*np.exp(-b*(t-t0))+c+d*np.exp(-e*t-t0))
        data,cov = curve_fit(f,time,amp)     
        return (time, amp-f(time,*data))


    
    if Type == 'BPF':
        b , a = sig.butter(2,[Freqmin/fs,Freqmax/fs],btype='band')
        return (time,sig.filtfilt(b, a, amp))

            
    if Type == 'Poly':
        poly = np.polyfit(time,amp,10)
        fonc = np.poly1d(poly)
        return (time,amp-fonc(time))