#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import time
from math import *
import copy

from pyquaternion import Quaternion
from pathlib import Path

import pandas as pd

import csv


def save_to_csv(data_list, path_to_save):
    with open(path_to_save, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for data in data_list:
            wr.writerow([data])


def relative_translation(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]


def rotation_error(r1, r2):
    diff_r = Quaternion(r1.elements - r2.elements)
    return diff_r.degrees

def translation_error(p1, p2):
    return sqrt(pow(p1[0] - p2[0],2) + pow(p1[1] - p2[1],2) + pow(p1[2] - p2[2],2))

def traj_translation_error(trj1, gt_traj):
    gt_index = 0
    errors = []
    for p in trj1:
        while gt_index < len(gt_traj) and p.timestamp > gt_traj[gt_index].timestamp:
            gt_index = gt_index + 1
        
        if gt_index < len(gt_traj):
            # print('P timestamp: ', p.timestamp, '  GT timestamp: ', gt_traj[gt_index].timestamp)
            # print('Time diff : ', p.timestamp - gt_traj[gt_index].timestamp)
            errors.append(translation_error(p.position, gt_traj[gt_index].position))

    return errors

def traj_relative_translation_error(trj1, gt_traj):
    gt_index = 0
    gt_last_index = 0
    errors = []

    last_p = None

    for p in trj1:
        while gt_index < len(gt_traj) and p.timestamp > gt_traj[gt_index].timestamp:
            gt_index = gt_index + 1
        
        if gt_index < len(gt_traj):
            # print('P timestamp: ', p.timestamp, '  GT timestamp: ', gt_traj[gt_index].timestamp)
            # print('Time diff : ', p.timestamp - gt_traj[gt_index].timestamp)
            if last_p != None:
                relative_trans = relative_translation(p.position,last_p.position)
                relative_trans_gt = relative_translation(gt_traj[gt_index].position, gt_traj[gt_last_index].position)

                errors.append(translation_error(relative_trans, relative_trans_gt))

            last_p = p
            gt_last_index = gt_index

    return errors


def translation(p, t):
    return np.array([p[0]-t[0], p[1]-t[1], p[2]-t[2]])

def distance(p1, p2):
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def error_log(errors):
    import statistics
    # print("    Means        Max        min")
    print('Mean : ', statistics.mean(errors), '    std dev  :  ',  statistics.stdev(errors), '    Max  :  ',  max(errors), '    Min  :  ',  min(errors))


class loc_point:
    def __init__(self):
        self.timestamp = 0
        self.position = None
        self.orientation = None


colors = ['seagreen', 'purple', 'yellow', 'lightcoral']
f, ax = plt.subplots()
f.set_size_inches(6.0, 5.0)
plt.subplots_adjust(left=0.16, right=0.95, top=0.90, bottom=0.15)
#f.suptitle('ORB2-SLAM with Euroc dataset')

#######################
###Load Ground Truth###
#######################
gt_path = '/home/ao/Projects/ORB_SLAM2/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv'

offset_flag = True
offset_p = [] #np.array()
offset_r = Quaternion()
offset_point = None
conj_offset_r = Quaternion()
ground_truth_traj = []



gtxs = []
gtys = []

with open(gt_path) as f:
    lines = f.readlines()
    for line in lines:
        content = [float(x) for x in line.split(',')]

        new_p = loc_point()

        new_p.timestamp = content[0] / (1.0 * 10e8)

        new_p.position = np.array([content[1], content[2], content[3]])
        new_p.orientation = Quaternion(content[4], content[5], content[6], content[7])

        if offset_flag:
            offset_point = copy.copy(new_p)
            conj_offset_r = offset_point.orientation.conjugate
            offset_flag = False
        #### Eliminate the initial offset
        new_p.position = translation(new_p.position, offset_point.position)
        new_p.position = conj_offset_r.rotate(new_p.position)

        new_p.orientation = Quaternion(new_p.orientation.elements - offset_point.orientation.elements)

        ### coordination transform
        new_p.position = np.array([new_p.position[1], -new_p.position[0], new_p.position[2]])

        ground_truth_traj.append(new_p)

        gtxs.append(new_p.position[0])
        gtys.append(new_p.position[1])


print("gt size : ", len(gtxs))
up = len(gtxs) / 4
time_stamp_up = ground_truth_traj[up].timestamp
ax.plot(gtxs[:up], gtys[:up], linewidth=2, color='black', label='Ground Truth')

###############################
###Load Experimental Results###
###############################
all_trajs = []
labels = ['None', 'Low', 'Medium', 'High']
dir_path = '/home/ao/Projects/ORB_SLAM2/environment_impacts/mh03_timing/'
index = 0
for txt_name in sorted(os.listdir(dir_path)):
    xs = []
    ys = []
    test_traj = []
    print(txt_name)
    with open(dir_path+txt_name) as f:
        lines = f.readlines()
        for line in lines:
            content = [float(x) for x in line.split()]

            new_p = loc_point()
            new_p.timestamp = content[0]
            new_p.position = np.array([content[1], content[2], content[3]])
            new_p.orientation = Quaternion(content[4], content[5], content[6], content[7])

            test_traj.append(new_p)
            if (content[0] > time_stamp_up):
                break
            xs.append(new_p.position[0])
            ys.append(new_p.position[1])
    # up = len(xs) / 4
    ax.plot(xs, ys, linewidth=2, color=colors[index], label=labels[index], path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
 #   ax.scatter(xs, ys, color=colors[index], label=txt_name)
    print("traj size : ", len(xs))
    index = index + 1


#ax.set_xlim([9.0, 13.0])
ax.set_ylim([-1.2, 2])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('X-axis position (m)',fontsize=16)
plt.ylabel('Y-axis position (m)',fontsize=16)
plt.legend(loc='upper left', fontsize=14)
plt.savefig('trajs.eps', format='eps')
plt.show()
exit(0)
