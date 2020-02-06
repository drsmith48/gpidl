#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:06:20 2019

@author: drsmith
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pims.cine import Cine

shotlist = 'shotlist_v01.csv'
cameradir= Path('/p/nstxcam-archive/Phantom710-9205/2010')

df = pd.read_csv(shotlist)
df = df.set_index('category')
df = df.set_index('shot', append=True)

total_frames = 0
nskip = 20*15
framepool = None
frames = np.empty((20000,64,80), dtype=np.uint16)
iframe = 0
noffset = 0


for cat, catdf in df.groupby(level=0):
    catdf = catdf.reset_index(level=0, drop=True)
    print('Category {} with {} entries'.format(cat, len(catdf)))
    cat_frames = 0
    for shot, row in catdf.iterrows():
        cine_filename = cameradir / 'nstx_5_{:d}.cin'.format(np.int(shot))
        assert(cine_filename.exists())
        framesequence = Cine(cine_filename)
        assert(framesequence.frame_rate==397660)
        # calculate discharges times for each frame
        # list of 2-tuples
        delta = [(ts[0]-framesequence.trigger_time['datetime'],
                  ts[1]-framesequence.trigger_time['second_fraction'])
                 for ts in framesequence.frame_time_stamps]
        shottimes = np.array([d[0].seconds + d[0].microseconds/1e6 + d[1]
                              for d in delta])*1e3
        assert(row.start>shottimes.min())
        assert(row.stop<shottimes.max())
        validtimes = shottimes[(shottimes>row.start) & (shottimes<row.stop)]
        validindices, = np.nonzero((shottimes>row.start) & (shottimes<row.stop))
        poolindices = validindices[noffset::nskip]
        for i in poolindices:
            frames[iframe,:,:] = framesequence[i]
            iframe += 1
        data = {'category':np.array([cat]*poolindices.size, dtype=np.int),
                'shot':np.array([shot]*poolindices.size, dtype=np.int),
                'iframe':poolindices,
                'shottime':shottimes[poolindices],
                'tstart':np.array([row.start]*poolindices.size),
                'tstop':np.array([row.stop]*poolindices.size),
                }
        if framepool is None:
            framepool = pd.DataFrame(data)
        else:
            framepool = pd.concat([framepool, pd.DataFrame(data)], ignore_index=True)
        print('  {} nframes {} valid fr. {} fr pool {} {}'.
              format(shot, framesequence.image_count, validtimes.size,
                     validtimes.size//nskip, poolindices.size))
        cat_frames += validtimes.size
    print('Category {} frames: {}'.format(cat, cat_frames))
    total_frames += cat_frames

print('Total frames: {}'.format(total_frames))
frames = frames[:iframe,...]
assert(frames.shape[0]==len(framepool))

filename = 'frame_category_small.pickle'
with open(filename, 'wb') as f:
    pickle.dump({'frames':frames, 'frameinfo':framepool}, f)
