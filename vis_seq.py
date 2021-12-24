
#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import copy

from pyquaternion import Quaternion
from pathlib import Path

import pandas as pd

import csv
import seaborn as sb
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

fig = figure()
ax = axes()
hold(True)
fig.set_size_inches(5.5, 4.0)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)



data = [
    [0.9,0.5,0.4,0.5,0.8],
    [0.8,0.5,0.4,0.5,0.6],
    [0.6,0.9,0.5,0.9,0.9],
    [0.4,0.7,0.8,0.9,0.4],
    [0.5,0.7,0.8,0.9,0.4]
]

data = np.asarray(data, dtype=np.float32)

ax.set_yticklabels(['mh_01_easy', 'mh_02_easy', 'mh_03_medium', 'mh_04_difficult', 'mh_05_difficult'], fontsize=14)
ax.set_yticks([0, 1, 2, 3, 4])

ax.set_xticklabels(['Dura.', 'Len.', 'Sce.', 'Trans.', 'Rotat.'], fontsize=14)
ax.set_xticks([0, 1, 2, 3, 4])
plt.imshow(data,cmap = 'Blues', vmin=0.0, vmax=1.0)

def rect(pos):
    r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor="k", linewidth=2)
    plt.gca().add_patch(r)

x,y = np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))
m = np.c_[x[data.astype(bool)],y[data.astype(bool)]]
for pos in m:
    rect(pos)

plt.colorbar(fontsize=14)
plt.show()
plt.savefig('seq_fea.eps', format='eps')
