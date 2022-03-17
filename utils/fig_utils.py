from turtle import width
import numpy as np
import pandas as pd
from scipy import signal as sig
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# plt.rc('text', usetex=True)
plt.rc('font', size=14)
plt.rc('font', family='sans-serif')
plt.rc('lines', linewidth=1)
plt.rc('axes', linewidth=1, labelsize=14)
plt.rc('legend', fontsize=14)

iii = 0
savepdf = True
def plot_heatmap_maxfrequency(X, Y, Z, clim = (None, None),  title = 'None'):
    global iii
    if iii > 2 and savepdf:
        nn = int(np.sqrt(Z.shape[0]))
        plt.figure()
        plt.figure(figsize=(6,5))
        plt.imshow(Z.reshape(nn,nn), extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='jet', vmin=clim[0],vmax=clim[1])
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(title, fontsize=12)
        plt.colorbar(label='Max Frequency (GHz)')
        plt.tight_layout()
        plt.savefig('MAP.png', transparent = True, dpi = 800)


    fig = go.Figure(go.Heatmap(x = X, y = Y, z = Z,colorbar={"title": "Frequency (GHz)"}, colorscale='Jet', zmin = clim[0], zmax = clim[1]))

    fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )
    fig.update_layout(title=title,
                  yaxis={"title": 'y (mm)'},
                  xaxis={"title": 'x (mm)'},width = 700, height = 600,
                  )
    iii +=1
    return fig

def plot_signals(raw_signal, filterd_signal, psd, winsize = 0.2, overlapsize = 0.1, fmax = 100, x=0, y=0):
    global iii
    global savepdf

    if iii > 2 and savepdf:
        print("st")
        # gs_kw = dict(width_ratios=[1.8, 1], height_ratios=[1, 2])
        gs_kw = dict(width_ratios=[2.6, 1], height_ratios=[2, 1])
        fig_, axd = plt.subplot_mosaic([['upper left', 'right'],
                                    ['lower left', 'right']],
                                    gridspec_kw=gs_kw, figsize=(8.5, 5),
                                    constrained_layout=True)

            
        # fig_.suptitle('None')
        axd['upper left'].plot(raw_signal[0,:], raw_signal[1,:], 'k')
        axd['upper left'].set_ylabel(r'$\Delta R/R$ (-)')
        axd['lower left'].set_ylabel(r'$\Delta R/R$ (-)')
        axd['upper left'].set_xlabel(r'Time (ns)')
        axd['lower left'].set_xlabel(r'Time (ns)')
        axd['lower left'].set_title('Filtered signal')
        axd['upper left'].set_title('Raw signal')
        
        axd['upper left'].set_xlim(-0.2, 4)
        axd['lower left'].plot(np.array(filterd_signal[0]),np.array(filterd_signal[1]),'k')
        axd['right'].plot(np.array(psd[1]/psd[1].max()), np.array(psd[0]),'k')
        axd['right'].set_ylim(0, 100)
        axd['right'].set_xlabel("Norm. amplitude (-)")
        axd['right'].set_ylabel("Frequency (GHz)")
        plt.suptitle(f'Raw signal at point (x,y) = ({x:.3f},{y:.3f})mm', fontsize=10)
        plt.tight_layout()
        plt.savefig('OneSignal.pdf')
        plt.close('all')
        print('done')




    fig = make_subplots(
        rows=3, cols=2, column_widths=[0.8, 0.2],
        specs=[[{}, {"rowspan": 2}],
            [{}, None],
            [{"colspan": 2}, None]],
        subplot_titles=(f"Raw signal at point (x,y) = ({x:.3f},{y:.3f})mm","PSD", "Filterd signal", 'STFT'))
    fs = abs(1/(filterd_signal[0][1] - filterd_signal[0][0]))
    f, t, Zxx = sig.stft(filterd_signal[1], fs = fs, nperseg=int(winsize*fs), noverlap=int(overlapsize*fs))

    fig.add_trace(go.Scatter(x = raw_signal[0,:], y=raw_signal[1,:], line_color = 'black'), row = 1, col=1)
    fig.add_trace(go.Scatter(x = filterd_signal[0], y=filterd_signal[1],line_color = 'black'), row = 2, col=1)
    fig.add_trace(go.Scatter(y = psd[0], x= psd[1]/np.mean(psd[1]),line_color = 'black'), row = 1, col=2)
    fig.add_trace(go.Heatmap(x = t, y=f, z = np.abs(Zxx) / np.abs(Zxx).max()), row = 3, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text=r" $\text{Time} (ns)$", row=1, col=1)
    fig.update_xaxes(title_text=r" $\text{Time} (ns)$", row=2, col=1)
    fig.update_xaxes(title_text="Norm. amplitude (-)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (GHz)",range=[2, 100], row=1, col=3)

    # Update yaxis properties
    fig.update_yaxes(title_text="$\Delta R/R$", row=1, col=1)
    fig.update_yaxes(title_text="$\Delta R/R$", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (GHz)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency (GHz)",range=[0, fmax], row=3, col=1)


    fig.update_layout(showlegend=False, height = 600,margin={"r":0,"t":50,"l":0,"b":0})

    iii += 1
    return(fig)


def plot_signals_withoutstft(raw_signal, filterd_signal, psd, winsize = 0.2, overlapsize = 0.1, fmax = 100, x=0, y=0):
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.8, 0.2],
        specs=[[{}, {"rowspan": 2}],
            [{}, None]],
        subplot_titles=(f"Raw signal at point (x,y) = ({x:.3f},{y:.3f})mm","PSD", "Filterd signal"))
    # fs = abs(1/(filterd_signal[0][1] - filterd_signal[0][0]))
    # f, t, Zxx = sig.stft(filterd_signal[1], fs = fs, nperseg=int(winsize*fs), noverlap=int(overlapsize*fs))

    fig.add_trace(go.Scatter(x = raw_signal[0,:], y=raw_signal[1,:], line_color = 'black'), row = 1, col=1)
    fig.add_trace(go.Scatter(x = filterd_signal[0,:], y=filterd_signal[1],line_color = 'black'), row = 2, col=1)
    fig.add_trace(go.Scatter(y = psd[0], x= psd[1]/psd[0].max(),line_color = 'black'), row = 1, col=2)
    # fig.add_trace(go.Heatmap(x = t, y=f, z = np.abs(Zxx) / np.abs(Zxx).max()), row = 3, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text=r" $\text{Time} (ns)$", row=1, col=1)
    fig.update_xaxes(title_text=r" $\text{Time} (ns)$", row=2, col=1)
    fig.update_xaxes(title_text="Norm. amplitude (-)", row=1, col=2)
    # fig.update_xaxes(title_text="Frequency (GHz)",range=[2, 100], row=1, col=3)

    # Update yaxis properties
    fig.update_yaxes(title_text="$\Delta R/R$", row=1, col=1)
    fig.update_yaxes(title_text="$\Delta R/R$", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (GHz)", row=1, col=2)
    # fig.update_yaxes(title_text="Frequency (GHz)",range=[0, fmax], row=3, col=1)


    fig.update_layout(showlegend=False, height = 600,margin={"r":0,"t":50,"l":0,"b":0})
    return(fig)




if __name__ == "__main__":
    import utils_filt
    from tqdm import tqdm
    MEASURMENT_NAME = '/Users/nicolas/Documents/Mesure_neta/03/spot_sizefiner'

    Df_data = pd.read_pickle(f'{MEASURMENT_NAME}/log_file.pkl')
    Df_datatxtfile = pd.read_csv(f'{MEASURMENT_NAME}/metadata.csv')
    signal = np.load(f'{MEASURMENT_NAME}/full_data.npy')
    print(signal.shape)


    t = signal[0,0,:]
    index = t>-0.2
    signal = signal[:,:,index]
    t = signal[0,0,:]
    tmax = 2

    signal[:,1,:] = signal[:,1,:] - np.mean(signal[:,1,np.logical_and(t>-0.2, t<-0.1)], axis = -1)[:,None]
    signal[:,1,:] = signal[:,1,:]/ np.max(signal[:,1,:], axis=1)[:, None]

    dt = abs(t[10]-t[11])
    nfft = 2**14
    freq =np.arange(nfft)/dt/nfft
    FREQMAX = 100
    FREQMAX_IDX = freq < FREQMAX

    signal_filt = np.asarray([utils_filt.filt(signal[idx, 0, t<tmax], signal[idx, 1, t<2], tmax) for idx in tqdm(range(len(Df_data)))])
    PSD = np.array([np.abs(np.fft.fft(signal_filt[idx,1], n = nfft)) for idx in range(len(Df_data))])[:,FREQMAX_IDX]
    freq = freq[FREQMAX_IDX]

    PSDmaxIDX = np.argmax(PSD, axis=1)
    PSDmax = freq[PSDmaxIDX]


    fig = plot_heatmap_maxfrequency(np.cumsum(Df_data['dx']), np.cumsum(Df_data['dy']), PSDmax)


    fig = plot_signals(signal[10,...],signal_filt[10,:], (freq, PSD[10,:]))
    fig.show()
    print(PSDmax.shape)
    print(PSD.shape)
    print('done')