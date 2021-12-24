#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import copy

import matplotlib.patheffects as pe

from pyquaternion import Quaternion
from pathlib import Path

import pandas as pd

import csv
import seaborn as sb

from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

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

# def traj_absolute_error(trj1, gt_traj):
#     gt_index = 0vehicle
#     last_p = None

#     cur_ATE_diff = [0, 0, 0]
#     cur_ATE = 0
#     for p in trj1:
#         while gt_index < len(gt_traj) and p.timestamp > gt_traj[gt_index].timestamp:
#             gt_index = gt_index + 1
        
#         if gt_index < len(gt_traj):

#             if last_p != None:
#                 relative_trans = relative_translation(p.position,last_p.position)
#                 relative_trans_gt = relative_translation(gt_traj[gt_index].position, gt_traj[gt_last_index].position)

#                 # cur_ATE_diff = cur_ATE_diff + 
#                 errors.append(translation_error(relative_trans, relative_trans_gt))

#             last_p = p
#             gt_last_index = gt_index

#     return errors


def traj_translation_error(trj1, gt_traj):
    gt_index = 0
    errors = []
    for p in trj1:
        while gt_index < len(gt_traj) and p.timestamp > gt_traj[gt_index].timestamp:
            gt_index = gt_index + 1
        
        if gt_index < len(gt_traj):
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
    print('Mean : ', statistics.mean(errors), '    std dev  :  ',  statistics.stdev(errors), '    Max  :  ',  max(errors), '    Min  :  ',  min(errors))


class loc_point:
    def __init__(self):
        self.timestamp = 0
        self.position = None
        self.orientation = None


colors = sb.color_palette('colorblind')

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
colors = sb.xkcd_palette(colors)
colors[0] = 'lightcoral'
fig = figure()
ax = axes()
hold(True)
fig.set_size_inches(12, 5.0)
plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.15)
#f.suptitle('ORB2-SLAM with Euroc dataset')

###########################
###Load All Ground Truth###
###########################
gt_dir = '/home/ao/Projects/ORB_SLAM2/ground_truth/'

gt_trajs = []
offset_p = [] #np.array()
offset_r = Quaternion()
offset_point = None
conj_offset_r = Quaternion()
ground_truth_traj = []


sequence_index = 0
sequence_labels = ['mh01', 'mh02', 'mh03', 'mh04', 'mh05']
gt_trajs = {'mh01': None , 'mh02': None, 'mh03': None, 'mh04' : None, 'mh05': None}
gtxs = []
gtys = []
for gt_csv_name in sorted(os.listdir(gt_dir)):
    print(gt_csv_name)
    ground_truth_traj = []
    offset_flag = True
    with open(gt_dir+gt_csv_name) as f:
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

    gt_trajs[sequence_labels[sequence_index]] = ground_truth_traj
    print('s lable : ', sequence_labels[sequence_index], ' csv file : ', gt_csv_name)
    sequence_index = sequence_index + 1
    # gt_trajs.append(ground_truth_traj)

print('--------------Loading End-----------------')
###################################
###Load All Experimental Results###
###################################

intensity_labels = ['None', 'Low', 'Medium', 'High']
traj_dict = {'None': [], 'Low': [], 'Medium': [], 'High': []}
dir_path = '/home/ao/Projects/ORB_SLAM2/environment_impacts/'
sequence_index = 0
for dir_name in sorted(os.listdir(dir_path)):
    print(dir_name)
    intensity_index = 0
    all_trajs = []
    for txt_name in sorted(os.listdir(dir_path+dir_name)):
        txt_path = dir_path + dir_name + '/' + txt_name
        xs = []
        ys = []
        test_traj = []
        with open(txt_path) as f:
            lines = f.readlines()
            for line in lines:
                content = [float(x) for x in line.split()]

                new_p = loc_point()
                new_p.timestamp = content[0]
                new_p.position = np.array([content[1], content[2], content[3]])
                new_p.orientation = Quaternion(content[4], content[5], content[6], content[7])

                test_traj.append(new_p)

                xs.append(new_p.position[0])
                ys.append(new_p.position[1])
        print('test txt :', txt_path, ' gt sequence label: ', sequence_labels[sequence_index], "  intensity index : ", intensity_index)
        print('test length : ', len(test_traj), '   gt length ; ' , len(gt_trajs[sequence_labels[sequence_index]]))
        rela_errors = traj_relative_translation_error(test_traj, gt_trajs[sequence_labels[sequence_index]])
        error_log(rela_errors)
        traj_dict[intensity_labels[intensity_index]].append(rela_errors)
        intensity_index = intensity_index + 1
        all_trajs.append(test_traj)

    sequence_index = sequence_index + 1


positions_list = []
positions_1 = [1,2,3,4,5]
positions_list.append(positions_1)
positions_2 = [x+6 for x in positions_1]
positions_list.append(positions_2)
positions_3 = [x+6 for x in positions_2]
positions_list.append(positions_3)
positions_4 = [x+6 for x in positions_3]
positions_list.append(positions_4)

ii = 0
for key in intensity_labels:
    print(key)
    value = traj_dict[key]
    print(len(value))
    medianprops = dict(color="darkslategray",linewidth=1.5)
    # for tjj in value:
        # print('@@',tjj)
    # bp0 = plt.boxplot(rela_errors_array, showfliers=False, vert=True,  # vertical box alignment
    bp = plt.boxplot(value, positions = positions_list[ii], showfliers=False, vert=True,  # vertical box alignment
                     patch_artist=True, # fill with color)
                     medianprops=medianprops,
                     capprops = dict(linestyle='-',linewidth=2, color='black'), 
                     whiskerprops = dict(linestyle='-',linewidth=2, color='black'))   
    # position = [x+6 for x in position]
    ii = ii + 1
    # break


    color_index = 0
    for box in bp['boxes']:
    # change outline color
        box.set(color='black', linewidth=1.5)
    # change fill color
        box.set_facecolor(colors[color_index])
    # change hatch
        box.set(hatch = '/', lw=2.0)

        color_index = color_index + 1
    
    
#################
## Set legends ##
#################  

ax.plot([],[], color='lightcoral', label='mh_01_easy', linewidth=7, path_effects=[pe.Stroke(linewidth=10, foreground='black'), pe.Normal()])
ax.plot([],[], color=colors[1], label='mh_02_easy', linewidth=7, path_effects=[pe.Stroke(linewidth=10, foreground='black'), pe.Normal()])
ax.plot([],[], color=colors[2], label='mh_01_medium', linewidth=7, path_effects=[pe.Stroke(linewidth=10, foreground='black'), pe.Normal()])
ax.plot([],[], color=colors[3], label='mh_01_difficult', linewidth=7, path_effects=[pe.Stroke(linewidth=10, foreground='black'), pe.Normal()])
ax.plot([],[], color=colors[4], label='mh_05_difficult', linewidth=7, path_effects=[pe.Stroke(linewidth=10, foreground='black'), pe.Normal()])

ax.set_xticklabels(['None', 'Low', 'Medium', 'High'])
ax.set_xticks([3, 9, 15, 21])

#################

x_tick_pos = [3, 9, 15, 21]
xticks = ['None', 'Low', 'Medium', 'High', 'None', 'Low', 'Medium', 'High', 'None', 'Low', 'Medium', 'High', 'None', 'Low', 'Medium', 'High', 'None', 'Low', 'Medium', 'High', 'None', 'Low', 'aaa']

# ax.set_xticklabels([y+1 for y in x_tick_pos], ['None', 'Low', 'Medium', 'High'])


# ax.set_xticklabels(['None', 'Low', 'Medium', 'High'])

ax.set_xlim([0.0, 24.0])
# ax.set_ylim([-3.75, 22])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('Relative Pose Error', fontsize=16)
plt.xlabel('Resource Contention Intensity', fontsize=16)
plt.legend(loc='upper left',fontsize=12)
plt.savefig('overhead.eps', format='eps')
plt.show()
exit(0)
