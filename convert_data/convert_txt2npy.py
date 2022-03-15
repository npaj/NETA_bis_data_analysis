"""
Extract data from all text file to 3D numpy arry
"""

import os
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing
NB_CPU = multiprocessing.cpu_count()

import numpy as np
import pandas as pd

import argparse

NB_CPU = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument("--path",type = str, help="path to Measurement Folder")
args = parser.parse_args()

Df = pd.read_pickle(f'{args.path}/log_file.pkl')
print(Df)


def load(filename):
    '''Load text file'''
    try:
        return np.loadtxt(filename).T
    except:
        return(np.zeros((20000, 2)).T)


Data = np.array(Parallel(n_jobs=NB_CPU-1)(delayed(load)(f'{args.path}/{filename}') for idx, filename in enumerate(tqdm(Df['filename']))))

np.save(f'{args.path}/full_data.npy', Data)

print(f'Data saved to {args.path[:-5]}/full_data.npy')