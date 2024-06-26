from doctest import testfile
from socket import getaddrinfo
from turtle import position
import numpy as np
import scipy.stats as st
import os
import csv

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.use('TkAgg')

print(matplotlib.get_configdir())

# index for csv title
index_mode = 3
index_n_split_client = 5


def get_conf_interval(index, data, conf_rate):
    data_stat = []
    for i in range(len(index)):
        conf_interval_low, conf_interval_high = st.t.interval(conf_rate, len(
            data[i, 1:]) - 1, loc=np.mean(data[i, 1:]), scale=st.sem(data[i, 1:]))
        conf_mean = np.mean(data[i, 1:])
        data_stat.append([index[i], conf_interval_low,
                          conf_mean, conf_interval_high])
    return np.array(data_stat)


def get_cdf(data):
    counts, bin_edges = np.histogram(data, bins=len(data), density=True)
    dx = bin_edges[1] - bin_edges[0]
    cdf = np.cumsum(counts) * dx
    # label = np.sort(data)
    return bin_edges[1:], cdf


def get_data(filename, index):
    data = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            data.append(row[index])           # 选择某一列加入到data数组中
        return data


if __name__ == '__main__':
    filepath = os.path.dirname(__file__)
    application = ['yoho', 'mlp']
    input_data_len = [65536, 64 * 3072]
    service_time_test = [[], []]  # mode 0
    conf_rate = 0.99

    for app_id in range(len(application)):
        test1_client_file = 'emulator_yoho/measurements/results/' + \
            application[app_id]+'/testcase_1/client.csv'
        test2_client_file = 'emulator_yoho/measurements/results/' + \
            application[app_id]+'/testcase_2/client.csv'
        test3_client_file = 'emulator_yoho/measurements/results/' + \
            application[app_id]+'/testcase_3/client.csv'
        client_file = [test2_client_file, test3_client_file, test1_client_file]

        # store-and-forward test2, cdu = input_data_len, no caching and computing on network nodes
        # compute-and-forward test3, cdu = input_data_len, no CDU
        # CDU test1, cdu = CDU
        for i in range(len(client_file)):
            index = 0
            len_data = len(get_data(client_file[i], 0)) - 1
            while(index < len_data):
                # print(f"file {i}, index {index}, len_data {len_data}")
                epochs = int(get_data(client_file[i], 0)[index+1])
                mode = int(get_data(client_file[i], index_mode)[index+1])
                if i == 0:
                    cdu = input_data_len[app_id]
                else:
                    # n_split_client = 5
                    cdu = input_data_len[app_id] / \
                        int(get_data(client_file[i], 5)[index+1])
                service_time_epoch = [float(x)
                                      for x in get_data(client_file[i], -1)[index+1+1:index+epochs+1]]
                service_time_epoch.insert(0, cdu)
                service_time_test[app_id].append(service_time_epoch)
                index = index + epochs

    # codes for plot figures yoho
    with plt.style.context(['science', 'ieee']):
        fig_width = 6.5
        barwidth = 0.3
        bardistance = barwidth * 1.5
        colordict = {
            'store_forward': '#DDAA33',
            'compute_forward': '#009988',
            'compute_forward_cdu': '#0077BB',
            'orange': '#EE7733',
            'red': '#993C00',
            'blue': '#3340AD'
        }
        colorlist = ['#DDAA33', '#009988', '#0077BB', '#EE7733', '#993C00', '#3340AD']
        markerdict = {
            'store_forward': 'v',
            'compute_forward': 's',
            'compute_forward_cdu': 'o',
        }
        markerlist = ['v', 's', 'o']
        legendlist = ['Store-Forward', 'Compute-Forward', 'Optimal CDU']

        plt.rcParams.update({'font.size': 11})

        fig = plt.figure(figsize=(fig_width, fig_width / 1.618))
        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 4])
        ax_1 = fig.add_subplot(spec[0])
        ax_1.xaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
        ax_1.yaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)

        # ====================================================
        # Plot Yoho
        app_id=0
        tmp = service_time_test[app_id]
        tmp = [tmp[0], tmp[1], tmp[3]]
        service_time = np.array(tmp)
        service_time_conf = get_conf_interval(
            service_time[:, 0], service_time, conf_rate)
        # print(service_time)
        print(service_time_conf)

        for box_id in range(0, service_time_conf.shape[0]):
            box1 = ax_1.boxplot(service_time_conf[box_id, 1:]*1000, positions=np.arange(1) - bardistance*box_id, vert=False, widths=barwidth,
                                showfliers=True, showmeans=False, patch_artist=True,
                                boxprops=dict(
                                    color='black', facecolor=colorlist[box_id], lw=1),
                                medianprops=dict(color='black'),
                                capprops=dict(color='black'),
                                whiskerprops=dict(color='black'),
                                flierprops=dict(
                                    color=colorlist[box_id], markeredgecolor=colorlist[box_id], marker=markerlist[box_id], ms=4),
                                meanprops=dict(markerfacecolor='black', markeredgecolor='black'))
        ax_1.set_xticks(np.arange(2000, 3601, 200))
        ax_1.set_xlim([1800, 3800])
        ax_1.axes.xaxis.set_visible(True)
        ax_1.axes.yaxis.set_visible(False)

        ax_2 = fig.add_subplot(spec[1])
        ax_2.xaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
        ax_2.yaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
        for line_id in range(0, service_time_conf.shape[0]):
            bin_compute, cdf_compute = get_cdf(service_time[line_id, 1:]*1000)
            line1 = ax_2.plot(bin_compute, cdf_compute, color=colorlist[line_id], lw=1.2, ls='-',
                            marker=markerlist[line_id], ms=4, markerfacecolor='none', label=legendlist[line_id])
        ax_2.set_xlabel(r'Service Time $T_s$ of Yoho [$ms$]')
        ax_2.set_ylabel(r'Likelihood of Occurrence')
        ax_2.set_xticks(np.arange(2000, 3601, 200))
        ax_2.set_xlim([1800, 3800])
        ax_2.set_yticks(np.arange(0, 1.1, 0.2))

        plt.legend(loc='lower right', ncol=1, frameon=True)
        fig.subplots_adjust(hspace=0.02)
        fig_path = 'emulator_yoho/measurements/figures/ts_cdus_' + \
            application[app_id]+'.pdf'
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')

    # codes for plot figures yoho
    with plt.style.context(['science', 'ieee']):
        fig_width = 6.5
        barwidth = 0.3
        bardistance = barwidth * 1.5
        colordict = {
            'store_forward': '#DDAA33',
            'compute_forward': '#009988',
            'compute_forward_cdu': '#0077BB',
            'orange': '#EE7733',
            'red': '#993C00',
            'blue': '#3340AD'
        }
        colorlist = ['#DDAA33', '#009988', '#0077BB', '#EE7733', '#993C00', '#3340AD']
        markerdict = {
            'store_forward': 'v',
            'compute_forward': 's',
            'compute_forward_cdu': 'o',
        }
        markerlist = ['v', 's', 'o']
        legendlist = ['Store-Forward', 'Compute-Forward', 'Optimal CDU']

        plt.rcParams.update({'font.size': 11})

        fig = plt.figure(figsize=(fig_width, fig_width / 1.618))
        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 4])
        ax_1 = fig.add_subplot(spec[0])
        ax_1.xaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
        ax_1.yaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)

        # ====================================================
        # Plot MLP
        app_id=1
        tmp = service_time_test[app_id]
        tmp = [tmp[0], tmp[1], tmp[5]]
        service_time = np.array(tmp)
        service_time_conf = get_conf_interval(
            service_time[:, 0], service_time, conf_rate)
        # print(service_time)
        print(service_time_conf)

        for box_id in range(0, service_time_conf.shape[0]):
            box1 = ax_1.boxplot(service_time_conf[box_id, 1:]*1000, positions=np.arange(1) - bardistance*box_id, vert=False, widths=barwidth,
                                showfliers=True, showmeans=False, patch_artist=True,
                                boxprops=dict(
                                    color='black', facecolor=colorlist[box_id], lw=1),
                                medianprops=dict(color='black'),
                                capprops=dict(color='black'),
                                whiskerprops=dict(color='black'),
                                flierprops=dict(
                                    color=colorlist[box_id], markeredgecolor=colorlist[box_id], marker=markerlist[box_id], ms=4),
                                meanprops=dict(markerfacecolor='black', markeredgecolor='black'))
        ax_1.set_xticks(np.arange(2000, 3601, 200))
        ax_1.set_xlim([1800, 3800])
        ax_1.axes.xaxis.set_visible(True)
        ax_1.axes.yaxis.set_visible(False)

        ax_2 = fig.add_subplot(spec[1])
        ax_2.xaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
        ax_2.yaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
        for line_id in range(0, service_time_conf.shape[0]):
            bin_compute, cdf_compute = get_cdf(service_time[line_id, 1:]*1000)
            line1 = ax_2.plot(bin_compute, cdf_compute, color=colorlist[line_id], lw=1.2, ls='-',
                            marker=markerlist[line_id], ms=4, markerfacecolor='none', label=legendlist[line_id])
        ax_2.set_xlabel(r'Service Time $T_s$ of MLP [$ms$]')
        ax_2.set_ylabel(r'Likelihood of Occurrence')
        ax_2.set_xticks(np.arange(2000, 3601, 200))
        ax_2.set_xlim([1800, 3800])
        ax_2.set_yticks(np.arange(0, 1.1, 0.2))

        plt.legend(loc='lower right', ncol=1, frameon=True)
        fig.subplots_adjust(hspace=0.02)
        fig_path = 'emulator_yoho/measurements/figures/ts_cdus_' + \
            application[app_id]+'.pdf'
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')