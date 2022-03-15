"""
Extract data from all text file to 3D numpy arry
"""

import os
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing


import numpy as np
import pandas as pd
import yaml
import torch

import argparse

NB_CPU = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument("--path",type = str, help="path to Measurement Folder")
args = parser.parse_args()

Df = pd.read_pickle(f'{args.path}/log_file.pkl')
print(Df)
print(Df)


def load(filename):
    '''Load text file'''
    try:
        file1 = open(filename, 'r')
    except:
        return None

    out = []
    for i, line in enumerate(file1):
        if i < 46:
            out.append(line)
        else:
            break
    file1 = "".join(out)
    file1 = file1.replace('# ', "")
    return yaml.safe_load(file1)


Data = np.array(Parallel(n_jobs=NB_CPU-1)(delayed(load)(f'{args.path}/{filename}') for idx, filename in enumerate(tqdm(Df['filename']))))
np.save(f'{args.path}/metadata.npy', Data)

print(f'MetaData saved to {args.path[:-5]}/metadata.npy')