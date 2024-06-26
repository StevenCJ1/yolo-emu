#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   analyze.py
@Time    :   2022/04/13 13:53:24
@Author  :   Jiakang Weng
@Version :   1.0
@Contact :   jiakang.weng@mailbox.tu-dresden.de
@License :   (C)Copyright 2021-2022
@Desc    :   None
'''

# here put the import lib
from cProfile import label
import csv
import os
from unicodedata import name
# import matplotlib.pyplot as plt
import numpy as np
# from torch import float32
import math
import scipy.stats as st

import matplotlib
import matplotlib.pyplot as plt


def get_data(filename, index):
    data = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            data.append(row[index])           # 选择某一列加入到data数组中
        return data


def get_conf_interval(index, data, conf_rate):
    data_stat = []
    for i in range(len(index)):
        conf_interval_low, conf_interval_high = st.t.interval(conf_rate, len(
            data[i, 1:]) - 1, loc=np.mean(data[i, 1:]), scale=st.sem(data[i, 1:]))
        conf_mean = np.mean(data[i, 1:])
        data_stat.append([index[i], conf_interval_low,
                          conf_mean, conf_interval_high])
    return np.array(data_stat)

# yoho
node_pro_yoho = []
node_cache_yoho = []
sum_node_pro_yoho = []
sum_node_cache_yoho = []
node_cdu_yoho = []

# mlp
node_pro_mlp = []
node_cache_mlp = []
sum_node_pro_mlp = []
sum_node_cache_mlp = []
node_cdu_mlp = []


dirname = os.path.dirname(__file__)
chunk_gap = 0.001

for filepath in ['emulator_yoho/measurements/testcase_1/proc_info_vnf0-s0.csv', 'emulator_yoho/measurements/testcase_1/proc_info_vnf1-s1.csv', 'emulator_yoho/measurements/testcase_1/proc_info_vnf2-s2.csv']:
    len_data = len(get_data(filepath, 0)) - 1

    proc_time = get_data(filepath, -1)[1:]

    cache_time = get_data(filepath, -2)[1:]

    index = 0

    pro_time_yoho = [] 
    cache_time_yoho = []
    sum_pro_time_yoho = []
    sum_cache_time_yoho = []
    cdu_list_yoho = []

    pro_time_mlp = [] 
    cache_time_mlp = []
    sum_pro_time_mlp = []
    sum_cache_time_mlp = []
    cdu_list_mlp = []

    while(index < len_data):
        mode = int(get_data(filepath,0)[index+1])
        epochs = int(get_data(filepath, 4)[index+1])
        n_split = int(get_data(filepath, 5)[index+1])
        chunk_num = int(get_data(filepath, 7)[index+1])
        if mode ==0: cdu_list_yoho.append(int(get_data(filepath, 8)[index+1]))
        elif mode == 1: cdu_list_mlp.append(int(get_data(filepath, 8)[index+1]))
        pro_mean_list = []  # 每组epochs的td平均值
        pro_sum_list = []  # the sum of td per epochs
        pro_var_list = []  # 每组epochs的td方差
        cache_mean_list = []  # 每组epochs的tc平均值
        cache_sum_list = []  # the sum of tc per epochs
        cache_var_list = []  # 每组epochs的tc方差
        # 获取每次epochs的tc,td
        for e_cur in range(epochs):  # handle cur_epochs -> n_split
            # 每个current_epochs的proc_data
            data_pro_list = proc_time[index:index + n_split]
            data_cache_list = cache_time[index:index + n_split]
            index = index + n_split
            # 转换str为float
            for i in range(n_split):
                data_pro_list[i] = float(data_pro_list[i])
                data_cache_list[i] = float(data_cache_list[i])

            pro_mean_list.append(np.mean(data_pro_list))
            pro_sum_list.append(np.sum(data_pro_list))
            pro_var_list.append(np.var(data_pro_list))
            cache_mean_list.append(np.mean(data_cache_list))
            cache_sum_list.append(np.sum(data_cache_list))
            cache_var_list.append(np.var(data_cache_list))
        if mode == 0: # yoho
            pro_time_yoho.append(pro_mean_list) # each epoch
            cache_time_yoho.append(cache_mean_list)
            sum_pro_time_yoho.append(pro_sum_list)
            sum_cache_time_yoho.append(cache_sum_list)
        elif mode == 1: #mlp
            pro_time_mlp.append(pro_mean_list) # each epoch
            cache_time_mlp.append(cache_mean_list)
            sum_pro_time_mlp.append(pro_sum_list)
            sum_cache_time_mlp.append(cache_sum_list)
    # yoho
    pro_time_yoho = np.array(pro_time_yoho) * 1000
    cache_time_yoho = np.array(cache_time_yoho) * 1000
    sum_pro_time_yoho = np.array(sum_pro_time_yoho) * 1000
    sum_cache_time_yoho = np.array(sum_cache_time_yoho) * 1000

    conf = get_conf_interval(np.arange(len(pro_time_yoho)), pro_time_yoho, 0.99)
    node_pro_yoho.append(conf[:, 2])
    conf = get_conf_interval(np.arange(len(cache_time_yoho)), cache_time_yoho, 0.99)
    node_cache_yoho.append(conf[:, 2])
    conf = get_conf_interval(np.arange(len(sum_pro_time_yoho)), sum_pro_time_yoho, 0.99)
    sum_node_pro_yoho.append(conf[:, 2])
    conf = get_conf_interval(np.arange(len(sum_cache_time_yoho)), sum_cache_time_yoho, 0.99)
    sum_node_cache_yoho.append(conf[:, 2])
    node_cdu_yoho.append(cdu_list_yoho)

    # mlp
    pro_time_mlp = np.array(pro_time_mlp) * 1000
    cache_time_mlp = np.array(cache_time_mlp) * 1000
    sum_pro_time_mlp = np.array(sum_pro_time_mlp) * 1000
    sum_cache_time_mlp = np.array(sum_cache_time_mlp) * 1000

    conf = get_conf_interval(
        np.arange(len(pro_time_mlp)), pro_time_mlp, 0.99)
    node_pro_mlp.append(conf[:, 2])
    conf = get_conf_interval(
        np.arange(len(cache_time_mlp)), cache_time_mlp, 0.99)
    node_cache_mlp.append(conf[:, 2])
    conf = get_conf_interval(
        np.arange(len(sum_pro_time_mlp)), sum_pro_time_mlp, 0.99)
    sum_node_pro_mlp.append(conf[:, 2])
    conf = get_conf_interval(
        np.arange(len(sum_cache_time_mlp)), sum_cache_time_mlp, 0.99)
    sum_node_cache_mlp.append(conf[:, 2])
    node_cdu_mlp.append(cdu_list_mlp)



print(node_pro)
# print(node_cache)

# for i in range(node_pro.shape[0]):
#     # print(node_cache[i, :] / node_cache[2, :])
#     print('-------------------')
#     print(node_pro_yoho[i, :] / node_pro_yoho[2, :])
#     print('===================')

# print(node_cache)

# print(node_cdu[0, :])

# plot figures
fig_width = 6.5
barwidth = 0.15
bardistance = barwidth * 1.2
colordict = {
    'compute_forward': '#0077BB',
    'store_forward': '#DDAA33',
    'store_forward_ia': '#009988',
    'orange': '#EE7733',
    'red': '#993C00',
    'blue': '#3340AD'
}
markerdict = {
    'compute_forward': 'o',
    'store_forward': 'v',
    'store_forward_ia': 's'
}

plt.rcParams.update({'font.size': 11})

fig = plt.figure(figsize=(fig_width, fig_width / 1.618))
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.grid(True, linestyle='--', which='major',
              color='lightgrey', alpha=0.5, linewidth=0.2)
ax.yaxis.grid(True, linestyle='--', which='major',
              color='lightgrey', alpha=0.5, linewidth=0.2)
x_index = np.arange(len(node_cdu[0, :]))
bar1 = ax.bar(x_index - bardistance, node_pro[0, :],
              barwidth, color=colordict['compute_forward'], bottom=node_cache[0, :])
bar2 = ax.bar(x_index, node_pro[1, :],
              barwidth, color=colordict['compute_forward'], bottom=node_cache[1, :])
bar3 = ax.bar(x_index + bardistance, node_pro[2, :],
              barwidth, color=colordict['compute_forward'], bottom=node_cache[2, :])
bar4 = ax.bar(x_index - bardistance, node_cache[0, :],
              barwidth, color=colordict['store_forward'])
bar5 = ax.bar(x_index, node_cache[1, :],
              barwidth, color=colordict['store_forward'])
bar6 = ax.bar(x_index + bardistance, node_cache[2, :],
              barwidth, color=colordict['store_forward'])
ax.legend([bar1, bar2, bar3, bar4, bar5, bar6], [
          r'$t_{c, 1}$', r'$t_{c, 2}$', r'$t_{c, 3}$', r'$t_{r, 1}$', r'$t_{r, 2}$', r'$t_{r, 3}$'], loc='upper left', ncol=3)
ax.set_xlabel(r'data segment size ($CDU$)')
ax.set_ylabel(r'Time ($ms$)')
plt.xticks(range(len(node_cdu[0, :])), node_cdu[0, :], rotation=30)
plt.savefig('emulator_yoho/measurements/plot/tc_td_cdus.pdf',
            dpi=600, bbox_inches='tight')

fig = plt.figure(figsize=(fig_width, fig_width / 1.618))
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.grid(True, linestyle='--', which='major',
              color='lightgrey', alpha=0.5, linewidth=0.2)
ax.yaxis.grid(True, linestyle='--', which='major',
              color='lightgrey', alpha=0.5, linewidth=0.2)
x_index = np.arange(len(node_cdu[0, :]))
line1, = ax.plot(x_index - bardistance,
                 sum_node_pro[0, :], color=colordict['compute_forward'], ls='-')
line2, = ax.plot(x_index, sum_node_pro[1, :],
                 color=colordict['compute_forward'], ls='--')
line3, = ax.plot(x_index + bardistance,
                 sum_node_pro[2, :], color=colordict['compute_forward'], ls='-.')
line4, = ax.plot(x_index - bardistance,
                 sum_node_cache[0, :], color=colordict['store_forward'], ls='-')
line5, = ax.plot(
    x_index, sum_node_cache[1, :], color=colordict['store_forward'], ls='--')
line6, = ax.plot(x_index + bardistance,
                 sum_node_cache[2, :], color=colordict['store_forward'], ls='-.')
ax.legend([line1, line2, line3, line4, line5, line6], [
          r'$\sum^n t_{c, 1}$', r'$\sum^n t_{c, 2}$', r'$\sum^n t_{c, 3}$', r'$\sum^n t_{r, 1}$', r'$\sum^n t_{r, 2}$', r'$\sum^n t_{r, 3}$'], loc='upper left', ncol=3)
ax.set_xlabel(r'data segment size ($CDU$)')
ax.set_ylabel(r'Time ($ms$)')
plt.xticks(range(len(node_cdu[0, :])), node_cdu[0, :], rotation=30)
plt.savefig('emulator_yoho/measurements/plot/nc_nd_cdus.pdf',
            dpi=600, bbox_inches='tight')
