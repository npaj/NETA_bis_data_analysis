import os

import yaml
import numpy as np
import pandas as pd
from scipy import signal as sig

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import utils.fig_utils as fig_utils
import utils.utils_filt as utils_filt

from tqdm import tqdm
import argparse

MATHJAX_CDN = '''
https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/
MathJax.js?config=TeX-MML-AM_CHTML'''

external_scripts = [
                    {'type': 'text/javascript',
                     'id': 'MathJax-script',
                     'src': MATHJAX_CDN,
                     },
                    ]



parser = argparse.ArgumentParser()
parser.add_argument("--path",type = str, help="path to Measurement Folder")
args = parser.parse_args()

Df_data = pd.read_pickle(os.path.join(args.path,'log_file.pkl'))
Df_datatxtfile = np.load(os.path.join(args.path,'metadata.npy'), allow_pickle=True)
signalRAW = np.load(os.path.join(args.path,'full_data.npy'))

# styles
table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}


app = dash.Dash(__name__,external_scripts=external_scripts)
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }}
app.title = "Fast Analysis"
server = app.server

###############
#Parameters
###############

#Filtering
FiltType = 'BPF'
Tmin = -0.2
Tmax = 2
Freqmin = 20
Freqmax = 100
threshold = 10

#FFT
nfft = 2**14
Freqmax_fft = 100

#STFT
winsize = 0.5
overlap = 0.4
cmin = Freqmin
cmax = Freqmax

#heamp2
winsize_timemap = .05
time_timemap = 1
cmin_time = cmin
cmax_time = cmax

Current_point = 0

Params = ['params_filt_type', 'params_filt_min', 
        'params_filt_max', 'params_filt_minfreq','params_filt_maxfreq',
        'params_fft_nfft', 'params_fft_max','params_stft_win','params_stft_overlap', 'params_stft_cmin', 'params_stft_cmax', "params_stft_cmin_2", 'params_stft_cmax_2', 'threshold']

##############
# Process signals
t = signalRAW[0,0,:]
index = t>Tmin
signal = signalRAW[:,:,index]
t = signal[0,0,:]

signal[:,1,:] = signal[:,1,:] - np.mean(signal[:,1,np.logical_and(t>-0.2, t<-0.1)], axis = -1)[:,None]
signal[:,1,:] = signal[:,1,:]/ np.max(signal[:,1,:], axis=1)[:, None]

dt = abs(t[10]-t[11])
FS =  1/dt
freq =np.arange(nfft)/dt/nfft
FREQMAX_IDX = freq < Freqmax

signal_filt = np.asarray([utils_filt.filt(signal[idx, 0, t<Tmax], signal[idx, 1, t<Tmax], 0, FiltType) for idx in tqdm(range(len(Df_data)))])
PSD = np.array([np.abs(np.fft.fft(signal_filt[idx,1], n = nfft)) for idx in range(len(Df_data))])[:,FREQMAX_IDX]
freq = freq[FREQMAX_IDX]
PSD[np.max(PSD/np.mean(PSD), axis=1)<threshold, 0]=100
PSDmaxIDX = np.argmax(PSD, axis=1)
PSDmax = freq[PSDmaxIDX]



fig1 = fig_utils.plot_heatmap_maxfrequency(np.cumsum(Df_data['dx']), np.cumsum(Df_data['dy']), PSDmax, clim = [cmin, cmax])
fig2 = fig_utils.plot_signals(signal[Current_point,...],signal_filt[Current_point,:], (freq, PSD[Current_point,:]))

##############


##### Layout
app.layout = html.Div(
    className="",
    children=[
        html.Div(
            className="header",
            children=[

                html.H2("Neta bis Signal analysis", style={"color": "white"}),
 
            ],
            style={"backgroundColor": "rgb(2,21,70)", "textAlign": "center"}
        ),
        
    html.Div(
        className='tot',
            children=[
                html.Div(
                    className="two-thirds column alpha",
                    children=[
                    html.Div(
                        className="row",
                        children=[
                            html.Div(
                                className="two-thirds column alpha",
                                children=[
                                    html.Div(className="row",children = [
                                    dcc.Graph(id='map', figure = fig1),  
                                    ])
                                ]
                            ),
                            html.Div(
                                className="one-third column omega",
                                children=[
                                    html.H4(id='params',
                                        children=['Parameters']),

                                    html.Div(
                                    className="one-third column alpha",
                                    children=[
                                        html.H6(id='paramsFilt',
                                            children=['Filtering']),
                                        html.Label('Type'),
                                        dcc.Dropdown(id = 'params_filt_type',
                                            options=[
                                                {'label': 'Bande filter', 'value': 'BPF'},
                                                {'label': u'polyfit', 'value': 'Poly'},
                                                {'label': 'exp polyfit', 'value': 'PolyExp'}
                                            ],
                                            value='BPF'
                                        ),
                                        html.Br(),
                                        html.Label('Min time (ns)'),
                                        dcc.Input(id = 'params_filt_min', value=str(Tmin), type='text'),
                                        html.Br(),
                                        html.Label('Max time (ns)'),
                                        dcc.Input(id = 'params_filt_max', value=str(Tmax), type='text'),
                                        html.Br(),
                                        html.Label('Min freq (GHz)'),
                                        dcc.Input(id = 'params_filt_minfreq', value=str(Freqmin), type='text'),
                                        html.Br(),
                                        html.Label('Max freq (GHz)'),
                                        dcc.Input(id = 'params_filt_maxfreq', value=str(Freqmax), type='text'),
                                        html.Br(),
                                        html.Label('Threshold'),
                                        dcc.Input(id = 'threshold', value=str(threshold), type='text'),
                                        ]),
                                        html.Div(
                                    className="one-third column omega",
                                        children=[
                                        html.H6(id='paramsFFT',
                                                children=['FFT']),

                                        html.Label('nfft (2**n)'),
                                        dcc.Input(id = 'params_fft_nfft', value=str(np.log2(nfft)), type='text'),
                                        html.Br(),
                                        html.Label('Max freq (GHz)'),
                                        dcc.Input(id = 'params_fft_max', value=str(Freqmax_fft), type='text'),


                                        html.H6(id='paramsSTFT',
                                                children=['STFT']),

                                        html.Label('win size (ns)'),
                                        dcc.Input(id = 'params_stft_win', value=str(winsize), type='text'),
                                        html.Br(),
                                        html.Label('overlap (ns)'),
                                        dcc.Input(id = 'params_stft_overlap', value=str(overlap), type='text'),
                                        html.Br(),
                                        html.Label('zmin (GHz)'),
                                        dcc.Input(id = 'params_stft_cmin', value=str(cmin), type='text'),
                                        html.Br(),
                                        html.Label('zmax (GHz)'),
                                        dcc.Input(id = 'params_stft_cmax', value=str(cmax), type='text'),
                                        html.Button(id='setbutton', n_clicks=0, children='Set Params')

                                ])

                                ]
                            ) 
                        ]
                    ),
                    html.Div(
                        className="row",
                        children=[
                        
                        dcc.Graph(id='signalplot', figure =fig2, style={
                            "width": "100%"})
                        ]
                    )
                ]),
                html.Div(
                    className="one-third column omega",
                    children=[
                        html.Div(className="row",children = [
                                    dcc.Graph(id='map2', figure = fig1),
                        html.Br(),
                        html.Label('time (ns)'),
                        dcc.Input(id = 'params_stft_time_2', value=str(winsize_timemap), type='text'),
                        html.Label('win size (ns)'),
                        dcc.Input(id = 'params_stft_win_2', value=str(time_timemap), type='text'),
                        html.Label('zmin (GHz)'),
                        dcc.Input(id = 'params_stft_cmin_2', value=str(cmin_time), type='text'),
                        html.Label('zmax (GHz)'),
                        dcc.Input(id = 'params_stft_cmax_2', value=str(cmax_time), type='text'),
                        html.Label('Threshold'),
                        dcc.Input(id = 'threshold_2', value=str(threshold), type='text'),
                        html.Button(id='set-button_timemap', n_clicks=0, children='Set Params'),
                        html.H4(children=['Signal informations']),
                        html.H6(id='info', children=[yaml.safe_dump(Df_datatxtfile[Current_point],default_flow_style=False)])
                        ])
                    
                    ])    
            ]
        )
    ]
)



##### call back

@app.callback(Output('map', 'figure'),
                Input('setbutton', 'n_clicks'),
                [State(_p, 'value') for _p in Params])
def Update(n_clicks, filttype, *params):
    global Current_point
    global signal_filt
    global FiltType 
    global Tmin
    global Tmax
    global Freqmin 
    global Freqmax 

    #FFT
    global nfft
    global Freqmax_fft

    #STFT
    global winsize
    global overlap
    global cmin
    global cmax
    global cmax_time
    global cmin_time
    global threshold
    FiltType = filttype

    Tmin, Tmax, Freqmin, Freqmax, nfft, Freqmax_fft, winsize, overlap, cmin, cmax, cmin_time, cmax_time,threshold =  np.array(params, dtype=float)
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

    signal_filt = np.asarray([utils_filt.filt(signal[idx, 0, np.logical_and(t>Tmin, t<Tmax)], signal[idx, 1, np.logical_and(t>Tmin, t<Tmax)], Type =FiltType, Freqmin=Freqmin, Freqmax = Freqmax) for idx in tqdm(range(len(Df_data)))])
    # print(signal_filt.shape)
    PSD = np.array([np.abs(np.fft.fft(signal_filt[idx,1], n = nfft)) for idx in range(len(Df_data))])[:,FREQMAX_IDX]
    print(PSD.shape)
    print('eeee')
    freq2 = freq[FREQMAX_IDX]
    # PSD[np.max(PSD/np.mean(PSD), axis=1)<threshold, 0]=100
    PSDmaxIDX = np.argmax(PSD, axis=1)
    PSDmax = freq2[PSDmaxIDX]
    



    fig1 = fig_utils.plot_heatmap_maxfrequency(np.cumsum(Df_data['dx']), np.cumsum(Df_data['dy']), PSDmax, title='Max frequency', clim=(cmin, cmax))
    # fig2 = fig_utils.plot_signals(signal[Current_point,...],signal_filt[Current_point,:], (freq, PSD[Current_point,:]), winsize=winsize, overlapsize=overlap,fmax = Freqmax_fft)


    return(fig1)



@app.callback([Output('signalplot', 'figure'),Output('info', 'children')],
                [Input('map', 'clickData')],
                )
def Update_map(clickData):
    idx =  clickData['points'][0]['pointNumber']
    
    global Current_point

    global FiltType 
    global Tmin
    global Tmax
    global Freqmin 
    global Freqmax 

    #FFT
    global nfft
    global Freqmax_fft

    #STFT
    global winsize
    global overlap

    Current_point = idx
    xx, yy = np.cumsum(Df_data['dx']), np.cumsum(Df_data['dy'])
    print(Current_point)
    print('-'*10)


    t = signalRAW[0,0,:]
    index = t>-0.2
    signal = signalRAW[Current_point,:,index].T
    print(signal.shape)
    t = signal[0,:]


    signal[1,:] = signal[1,:] - np.mean(signalRAW[Current_point,1,np.logical_and(signalRAW[Current_point,0,:]>-0.2, signalRAW[Current_point,0,:]<-0.1)], axis = -1)
    # signal[1,:] = signal[1,:]/ np.max(signal[1,:])

    dt = abs(t[10]-t[11])
    freq =np.arange(nfft)/dt/nfft
    FREQMAX_IDX = np.logical_and(freq < Freqmax, freq > Freqmin)

    signal_filt = utils_filt.filt(signal[ 0, np.logical_and(t>Tmin, t<Tmax)], signal[1,np.logical_and(t>Tmin, t<Tmax)], 0, FiltType, Freqmin, Freqmax) 
    PSD = np.abs(np.fft.fft(signal_filt[1], n = nfft))[FREQMAX_IDX]
    freq = freq[FREQMAX_IDX]
    fig2 = fig_utils.plot_signals(signal,signal_filt, (freq, PSD), winsize=winsize, overlapsize=overlap,fmax = Freqmax_fft, x=xx[idx], y=yy[idx])
    return(fig2,yaml.safe_dump(Df_datatxtfile[Current_point],default_flow_style=False))

@app.callback(Output('map2', 'figure'),
                [Input('set-button_timemap', 'n_clicks')],
                [State('params_stft_time_2', 'value'),State('params_stft_win_2', 'value'), State('params_stft_cmin_2', 'value'), State('params_stft_cmax_2', 'value'), State('threshold_2', 'value')]
                )
def Update_map(click,current_time, winsize, cmin2, cmax2, th):
    global signal_filt
    global threshold
    print(cmin2, cmax2)
    current_time = float(current_time)
    winsize = float(winsize)
    PSDMAX = np.zeros(len(Df_data))
    for idx in tqdm(range(len(Df_data))):
        t = signal_filt[idx, 0]
        IDX_TIME = np.logical_and(t>current_time-winsize/2, t<current_time+winsize/2)
        x = signal_filt[idx, 1][IDX_TIME]
        t = signal_filt[idx, 0][IDX_TIME]
        freq = (np.arange(nfft)*FS/nfft)
        X = np.abs(np.fft.fft(x*sig.windows.hann(len(x), False), nfft))[np.logical_and(freq> Freqmin, freq < Freqmax)]
        freq = freq[np.logical_and(freq> Freqmin, freq < Freqmax)]
        # if np.max(X)/np.mean(X)<float(th):
        #     X[0] = 100
        PSDmaxIDX = np.argmax(X)
        
        PSDMAX[idx] = freq[PSDmaxIDX]
        
    
    return(fig_utils.plot_heatmap_maxfrequency(np.cumsum(Df_data['dx']), np.cumsum(Df_data['dy']), PSDMAX, title = f'Max frequency at a given at {current_time:.3f} ns with {winsize:.3f} ns window size', clim = (float(cmin2), float(cmax2))))



if __name__ == '__main__':
    # app.run_server(debug=False)#, host='127.0.0.1',port=os.getenv("PORT", "8051"))
    app.run_server(debug=True)#, host='127.0.0.1',port=os.getenv("PORT", "8051"))