import numpy as np
import matplotlib.pyplot as plt
import utils.utils_filt as utils_filt
import pandas as pd
from tqdm import tqdm

# plt.rc('text', usetex=True)
plt.rc('font', size=11)
plt.rc('font', family='serif')
plt.rc('lines', linewidth=1)
plt.rc('axes', linewidth=1, labelsize=10)
plt.rc('legend', fontsize=10)
###########
# x20
###########

measurement = '13/Ice_x20scan'
signalRAW = np.load(f'{measurement}/full_data.npy')
Df_data = pd.read_pickle(f'{measurement}/log_file.pkl')


xx, yy = np.cumsum(np.array(Df_data['dx'])), np.cumsum(np.array(Df_data['dy']))
Tmin, Tmax, Freqmin, Freqmax, nfft, Freqmax_fft=  0, 3, 20, 100, 15, 100
nfft = int(2**nfft)


t = signalRAW[0,0,:]
index = t>-0.2
signal = signalRAW[:,:,index]
t = signal[0,0,:]


signal[:,1,:] = signal[:,1,:] - np.mean(signalRAW[:,1,np.logical_and(signalRAW[0,0,:]>-0.2, signalRAW[0,0,:]<-0.1)], axis = -1)[:,None]
# signal[:,1,:] = signal[:,1,:]/ np.max(signal[:,1,:], axis=1)[:, None]

dt = abs(t[10]-t[11])
freq =np.arange(nfft)/dt/nfft
FREQMAX_IDX = np.logical_and(freq < Freqmax, freq > Freqmin)

signal_filt = np.asarray([utils_filt.filt(signal[idx, 0, np.logical_and(t>Tmin, t<Tmax)], signal[idx, 1, np.logical_and(t>Tmin, t<Tmax)], Type ='BPF', Freqmin=Freqmin, Freqmax = Freqmax) for idx in tqdm(range(len(Df_data)))])
# print(signal_filt.shape)
PSD = np.array([np.abs(np.fft.fft(signal_filt[idx,1], n = nfft)) for idx in range(len(Df_data))])[:,FREQMAX_IDX]
print(PSD.shape)
print('eeee')
freq2 = freq[FREQMAX_IDX]



PSDmaxIDX = np.argmax(PSD, axis=1)
PSDmax = freq2[PSDmaxIDX]
print(PSDmax.shape)
plt.figure(figsize=(4,3))
plt.imshow(PSDmax.reshape(50,50), extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap='jet', vmin=20,vmax=50)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(label='Max Frequency (GHz)')
plt.tight_layout()
plt.savefig('x20scan.png', transparent = True, dpi = 800)


Tmin, Tmax, Freqmin, Freqmax, nfft, Freqmax_fft=  -0., 2, 2, 100, 14, 100
nfft = int(2**nfft)


t = signalRAW[0,0,:]
index = t>-0.2
signal = signalRAW[:,:,index]
t = signal[0,0,:]


signal[:,1,:] = signal[:,1,:] - np.mean(signalRAW[:,1,np.logical_and(signalRAW[0,0,:]>-0.2, signalRAW[0,0,:]<-0.1)], axis = -1)[:,None]
# signal[:,1,:] = signal[:,1,:]/ np.max(signal[:,1,:], axis=1)[:, None]

dt = abs(t[10]-t[11])
freq =np.arange(nfft)/dt/nfft
FREQMAX_IDX = np.logical_and(freq < Freqmax, freq > Freqmin)

signal_filt = np.asarray([utils_filt.filt(signal[idx, 0, np.logical_and(t>Tmin, t<Tmax)], signal[idx, 1, np.logical_and(t>Tmin, t<Tmax)], Type ='BPF', Freqmin=Freqmin, Freqmax = Freqmax) for idx in tqdm(range(len(Df_data)))])
# print(signal_filt.shape)
PSD = np.array([np.abs(np.fft.fft(signal_filt[idx,1], n = nfft)) for idx in range(len(Df_data))])[:,FREQMAX_IDX]
print(PSD.shape)
print('eeee')
freq2 = freq[FREQMAX_IDX]



PSDmaxIDX = np.argmax(PSD, axis=1)

PSDmax = freq2[PSDmaxIDX]
print(PSDmax.shape)
plt.figure(figsize=(4,3))
plt.imshow(PSDmax.reshape(50,50), extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap='jet')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(label='Max Frequency (GHz)')
plt.tight_layout()
plt.savefig('x20scanlow.png', transparent = True, dpi = 800)



measurement = '12/Scan_ice_4'#'16/scan_icex50'
aa = 46
signalRAW = np.load(f'{measurement}/full_data.npy')
Df_data = pd.read_pickle(f'{measurement}/log_file.pkl')


xx, yy = np.cumsum(np.array(Df_data['dx'])), np.cumsum(np.array(Df_data['dy']))
Tmin, Tmax, Freqmin, Freqmax, nfft, Freqmax_fft=  0, 3, 20, 100, 15, 100
nfft = int(2**nfft)


t = signalRAW[0,0,:]
index = t>-0.2
signal = signalRAW[:,:,index]
t = signal[0,0,:]


signal[:,1,:] = signal[:,1,:] - np.mean(signalRAW[:,1,np.logical_and(signalRAW[0,0,:]>-0.2, signalRAW[0,0,:]<-0.1)], axis = -1)[:,None]
# signal[:,1,:] = signal[:,1,:]/ np.max(signal[:,1,:], axis=1)[:, None]

dt = abs(t[10]-t[11])
freq =np.arange(nfft)/dt/nfft
FREQMAX_IDX = np.logical_and(freq < Freqmax, freq > Freqmin)

signal_filt = np.asarray([utils_filt.filt(signal[idx, 0, np.logical_and(t>Tmin, t<Tmax)], signal[idx, 1, np.logical_and(t>Tmin, t<Tmax)], Type ='BPF', Freqmin=Freqmin, Freqmax = Freqmax) for idx in tqdm(range(len(Df_data)))])
# print(signal_filt.shape)
PSD = np.array([np.abs(np.fft.fft(signal_filt[idx,1], n = nfft)) for idx in range(len(Df_data))])[:,FREQMAX_IDX]
print(PSD.shape)
print('eeee')
freq2 = freq[FREQMAX_IDX]



PSDmaxIDX = np.argmax(PSD, axis=1)

PSDmax = freq2[PSDmaxIDX]
print(PSDmax.shape)
plt.figure(figsize=(4,3))
plt.imshow(PSDmax.reshape(aa,aa), extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap='jet', vmin=20,vmax=50)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(label='Max Frequency (GHz)')
plt.tight_layout()
plt.savefig('x50scan.png', transparent = True, dpi = 800)


Tmin, Tmax, Freqmin, Freqmax, nfft, Freqmax_fft=  -0., 2, 2, 100, 14, 100
nfft = int(2**nfft)


t = signalRAW[0,0,:]
index = t>-0.2
signal = signalRAW[:,:,index]
t = signal[0,0,:]


signal[:,1,:] = signal[:,1,:] - np.mean(signalRAW[:,1,np.logical_and(signalRAW[0,0,:]>-0.2, signalRAW[0,0,:]<-0.1)], axis = -1)[:,None]
# signal[:,1,:] = signal[:,1,:]/ np.max(signal[:,1,:], axis=1)[:, None]

dt = abs(t[10]-t[11])
freq =np.arange(nfft)/dt/nfft
FREQMAX_IDX = np.logical_and(freq < Freqmax, freq > Freqmin)

signal_filt = np.asarray([utils_filt.filt(signal[idx, 0, np.logical_and(t>Tmin, t<Tmax)], signal[idx, 1, np.logical_and(t>Tmin, t<Tmax)], Type ='BPF', Freqmin=Freqmin, Freqmax = Freqmax) for idx in tqdm(range(len(Df_data)))])
# print(signal_filt.shape)
PSD = np.array([np.abs(np.fft.fft(signal_filt[idx,1], n = nfft)) for idx in range(len(Df_data))])[:,FREQMAX_IDX]
print(PSD.shape)
print('eeee')
freq2 = freq[FREQMAX_IDX]



PSDmaxIDX = np.argmax(PSD, axis=1)

PSDmax = freq2[PSDmaxIDX]
print(PSDmax.shape)
plt.figure(figsize=(4,3))
plt.imshow(PSDmax.reshape(aa,aa), extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap='jet')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(label='Max Frequency (GHz)')
plt.tight_layout()
plt.savefig('x50scanlow.png', transparent = True, dpi = 800)

plt.show()
