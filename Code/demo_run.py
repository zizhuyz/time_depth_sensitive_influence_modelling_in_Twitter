#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/03/2019 8:46 AM
# @Author  : Zizhu
# @File    : DemoRun.py
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cal_prob as cal
import models
import os
import Tkinter as tk



# %% data
interactions = pd.read_csv('factors1.txt', sep=',')
network = pd.read_csv('network1.txt', sep=',')

# model parameter
alpha_TC, alpha_TD, alpha_TDD = 1, 1, 1.2
time_frame, time_slot, N_steps = 300, 10, 60
depth_decay_factor = 0.2
p1, beta = 0.001, 0.5

# draw parameter
default_node_size, default_label_size, default_edge_width = 700, 20, 1.0
title_size, ax_label_size = 16, 12

# file path
current_folder = os.getcwd()
inputs_folder = current_folder+'/cal_inputs/'


def ps_path(folder, md, alpha, time_frame, time_slot, p1, beta):
    return folder+'%s_ps_a%s_t%s_s%s_p%s_b%s.pkl' % (str(md), str(alpha), str(time_frame), str(time_slot), str(p1), str(beta))


def g_path(folder, md, time_frame, time_slot, p1, beta):
    return folder+'%s_g_t%s_s%s_p%s_b%s.pkl' % (str(md), str(time_frame), str(time_slot), str(p1), str(beta))


c_pre_ps_path = ps_path(inputs_folder, 'TC', alpha_TC, time_frame, time_slot, p1, beta)
d_pre_ps_path = ps_path(inputs_folder, 'TD', alpha_TD, time_frame, time_slot, p1, beta)
dd_pre_ps_path = ps_path(inputs_folder, 'TDD', alpha_TDD, time_frame, time_slot, p1, beta)


c_G_path = g_path(inputs_folder, 'TC', time_frame, time_slot, p1, beta)
d_G_path = g_path(inputs_folder, 'TD', time_frame, time_slot, p1, beta)
dd_G_path = g_path(inputs_folder, 'TDD', time_frame, time_slot, p1, beta)


# %% get probabilities for each model
# TC
def inputs_TC():
    if not os.path.isfile(c_G_path):
        c_p_df, c_pre_ps, c_G = models.get_inputs_for_c_model(interactions, network, time_frame=time_frame, time_slot=time_slot,p1=p1, alpha=alpha_TC, beta=beta)
        cal.save_obj(c_pre_ps, c_pre_ps_path)
        nx.write_gpickle(c_G, c_G_path)
    else:
        c_G = nx.read_gpickle(c_G_path)
        c_p_df = nx.to_pandas_edgelist(c_G, source='u', target='v', nodelist=c_G.nodes)
        if not os.path.isfile(c_pre_ps_path):
            print('no pre ps, calculating')
            c_pre_ps = cal.get_pre_cal_const_ps(time_frame, time_slot, c_p_df, alpha=alpha_TC)
            cal.save_obj(c_pre_ps, c_pre_ps_path)
        else:
            print('just read pre cal data')
            c_pre_ps = cal.load_obj(c_pre_ps_path)
    return c_G, c_pre_ps


# TD
def inputs_TD():
    if not os.path.isfile(d_G_path):
        d_p_df, d_pre_ps, d_G = models.get_inputs_for_d_model(interactions, network, time_frame=time_frame, time_slot=time_slot,p1=p1, alpha=alpha_TD, beta=beta)
        cal.save_obj(d_pre_ps, d_pre_ps_path)
        nx.write_gpickle(d_G, d_G_path)
    else:
        d_G = nx.read_gpickle(d_G_path)
        d_p_df = nx.to_pandas_edgelist(d_G, source='u', target='v', nodelist=d_G.nodes)
        if not os.path.isfile(d_pre_ps_path):
            print('no pre ps, calculating')
            d_pre_ps = cal.get_pre_cal_decay_ps(time_frame, time_slot, d_p_df, alpha=alpha_TD)
            cal.save_obj(d_pre_ps, d_pre_ps_path)
        else:
            print('just read pre cal data')
            d_pre_ps = cal.load_obj(d_pre_ps_path)
    return d_G, d_pre_ps


# TDD
def inputs_TDD():
    if not os.path.isfile(dd_G_path):
        dd_p_df, dd_pre_ps, dd_G = models.get_inputs_for_d_model(interactions, network, time_frame=time_frame, time_slot=time_slot,p1=p1, alpha=alpha_TDD, beta=beta)
        cal.save_obj(dd_pre_ps, dd_pre_ps_path)
        nx.write_gpickle(dd_G, dd_G_path)
    else:
        dd_G = nx.read_gpickle(dd_G_path)
        dd_p_df = nx.to_pandas_edgelist(dd_G, source='u', target='v', nodelist=dd_G.nodes)
        if not os.path.isfile(dd_pre_ps_path):
            print('no pre ps, calculating')
            dd_pre_ps = cal.get_pre_cal_decay_ps(time_frame, time_slot, dd_p_df, alpha=alpha_TDD)
            cal.save_obj(dd_pre_ps, dd_pre_ps_path)
        else:
            print('just read pre cal data')
            dd_pre_ps = cal.load_obj(dd_pre_ps_path)
    return dd_G, dd_pre_ps


c_G, c_pre_ps = inputs_TC()
d_G, d_pre_ps = inputs_TD()
dd_G, dd_pre_ps = inputs_TDD()


# choose a seed node
root = list(c_G.nodes)[4]

# # plot bar and curve of the decay ps
# height = d_pre_ps[0.03799999999981096]
# yy = np.arange(1, 31)*10
# bar_pos = yy-5
# bar_width = 9
# plt.plot(bar_pos, height)
# plt.bar(bar_pos, height, width=bar_width)
# plt.show()


# %%
# show the whole network structure
position = {'a': np.array([ 0.9, -0.02344431]),
 'c': np.array([ 0.35,  1.5]),
 'd': np.array([-0.4,  0.45]),
 'j': np.array([ 0.35,  0.18045684]),
 'k': np.array([-0.1, -0.36]),
 'n': np.array([-1, -0.36])}


def draw_whole_network():
    fig = plt.figure(figsize=(10, 8))
    # set backgroud color
    fig.set_facecolor("beige")
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_title('The Influence Network', fontsize=title_size)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    # draw nodes
    nx.draw_networkx_nodes(c_G, position, node_size=default_node_size, node_color='silver')
    # draw edges
    nx.draw_networkx_edges(c_G, position, width=default_edge_width, edge_color='k', node_size=default_node_size, arrowsize=20)
    # node labels
    nx.draw_networkx_labels(c_G, position, font_size=default_label_size)
    # edge_labels
    e_labels_1 = dict([((u, v), float("{0:.3f}".format(d['prob']))) for u, v, d in c_G.edges(data=True)])
    e_labels_2 = dict([((u, v), 'p1') for u, v, d in c_G.edges(data=True) if d['prob'] <= p1])
    e_labels = e_labels_1.copy()
    e_labels.update(e_labels_2)
    nx.draw_networkx_edge_labels(c_G, position, edge_labels=e_labels, label_pos=0.4, font_size=12, ont_color='k')
    # plt.axis('off')
    plt.show()


def draw_diffusion(nd0, nd1, G, fig, nd_shape):
    # color the root to red and the successors of the root to skyblue
    # nodes: 4 statues
    # nd0 = set(nd0)  # influenced but influence has dropped to 0 after T0 time
    # nd1 = set(nd1)  # influenced and active
    nd2 = set()  # not influenced and have influenced and active neighbours
    for nd in nd1:
        nd2 = nd2.union(G.successors(nd))
    nd2 = nd2-nd0-nd1
    nd3 = set(G.nodes)-nd0-nd1-nd2  # not influenced and do not have influenced and active neighbours

    # edges: 2 status
    ed1 = [e for e in G.edges if e[0] in nd1]
    ed2 = list(set(G.edges) - set(ed1))
    edge_width_by_prob = []
    for e in ed1:
        the_p = G[e[0]][e[1]]['prob']
        edge_width_by_prob.append(2*default_edge_width if the_p < p1*10 else (the_p/p1)**(0.3)*2*default_edge_width)

    # draw subfigure
    fig.set_yticklabels([])
    fig.set_xticklabels([])

    nx.draw_networkx_nodes(G, pos=position, nodelist=nd0, node_size=default_node_size, node_color='green', node_shape=nd_shape)
    nx.draw_networkx_nodes(G, pos=position, nodelist=nd1, node_size=default_node_size, node_color='r', node_shape=nd_shape)
    nx.draw_networkx_nodes(G, pos=position, nodelist=nd2, node_size=default_node_size, node_color='skyblue', node_shape=nd_shape)
    nx.draw_networkx_nodes(G, pos=position, nodelist=nd3, node_size=default_node_size, node_color='silver', node_shape=nd_shape)
    nx.draw_networkx_labels(G, position, font_size=default_label_size)

    nx.draw_networkx_edges(G, pos=position, edgelist=ed2, width=default_edge_width, edge_color='k', node_size=default_node_size)
    nx.draw_networkx_edges(G, pos=position, edgelist=ed1, width=edge_width_by_prob, edge_color='r', node_size=default_node_size)


def draw_one_summary(c_detail, d_detail, dd_detail, ax1):
    """
    :param c_detail:  np array, the activated node at each time step, axis 1 are "step activated_node  depth"
    :param d_detail:
    :param dd_detail:
    :return:
    """
    # fig_sum, ax1 = plt.subplots()
    # # set backgroud color
    # fig_sum.set_facecolor("beige")
    ax1.set_ylabel('Influenced Time Step', fontsize=ax_label_size)
    ax1.set_xlabel('Depth to Root User', fontsize=ax_label_size)
    ax1.set_ylim(N_steps, -int(N_steps/10))
    ax1.set_yticks(range(-int(N_steps/10), N_steps+1, int(N_steps/10)))
    ax1.set_xlim(-0.3, 4.5)
    ax1.set_xticks([0, 1, 2, 3])
    if not c_detail.size == 0:
        ax1.plot(c_detail[:, 2], c_detail[:, 0], color='mediumturquoise', marker='o', markersize=16, linestyle='None', label='TC')
    if not d_detail.size == 0:
        ax1.plot(d_detail[:, 2], d_detail[:, 0], color='lightgreen', marker='s', markersize=12, linestyle='None', label='TD')
    if not dd_detail.size == 0:
        ax1.plot(dd_detail[:, 2], dd_detail[:, 0], color='darkorange', marker='H', markersize=10, linestyle='None', label='TDD')


def get_diffusion(c_history=np.empty((0,3), int), d_history=np.empty((0,3), int), dd_history=np.empty((0,3), int)):
    exe_time = 0
    c_rslt, c_detail = models.model('TC', c_pre_ps, c_G, time_frame, time_slot, N_steps, root, alpha=alpha_TC)
    d_rslt, d_detail = models.model('TD', d_pre_ps, d_G, time_frame, time_slot, N_steps, root, alpha=alpha_TD)
    dd_rslt, dd_detail = models.model('TDD', dd_pre_ps, dd_G, time_frame, time_slot, N_steps, root, alpha=alpha_TDD,
                                      depth_decay_factor=depth_decay_factor)
    # plot three models results in one figure
    fig2 = plt.figure(figsize=(20, 32))
    n_row,  n_column = 8, 4
    gs = gridspec.GridSpec(n_row, n_column)
    fig2.set_facecolor("beige")
    i,j,k = 0,0,0
    for index, row in c_rslt.iterrows():
        ax = plt.subplot(gs[i, 0])
        if i == 0:
            ax.set_title('Time Constant Model (TC) \n $%s$ post a message at time $t=%d$mins'%(list(row[1])[0], row[0]*time_slot), fontsize=title_size)
            draw_diffusion(set([]), row[1], c_G, ax, 'o')
        else:
            ax.set_title('Diffusion at time $t=%d$'%(row[0]), fontsize=title_size)
            draw_diffusion(row[2], row[1], c_G, ax, 'o')
        i += 1

    for index, row in d_rslt.iterrows():
        ax = plt.subplot(gs[j, 1])
        if j == 0:
            ax.set_title('Time Decay Model (TD) \n $%s$ post a message at time $t=%d$mins'%(list(row[1])[0], row[0]*time_slot), fontsize=title_size)
            draw_diffusion(set([]), row[1], d_G, ax, 's')
        else:
            ax.set_title('Diffusion at time $t=%d$'%(row[0]), fontsize=title_size)
            draw_diffusion(row[2], row[1], d_G, ax, 's')
        j += 1

    for index, row in dd_rslt.iterrows():
        ax = plt.subplot(gs[k, 2])
        if k == 0:
            ax.set_title('Time and Depth Decay Model (TDD) \n $%s$ post a message at time $t=%d$mins'%(list(row[1])[0], row[0]*time_slot), fontsize=title_size)
            draw_diffusion(set([]), row[1], dd_G, ax, 'H')
        else:
            ax.set_title('Diffusion at time $t=%d$'%(row[0]), fontsize=title_size)
            draw_diffusion(row[2], row[1], dd_G, ax, 'H')
        k += 1
    # plot the summary of the diffusion results of the three model
    ax1 = plt.subplot(gs[0:3, 3])
    ax1.set_title('One Diffusion Summary', fontsize=title_size)
    draw_one_summary(c_detail.values, d_detail.values, dd_detail.values, ax1)
    legend = ax1.legend(loc='lower right', fontsize=14)
    legend.get_frame().set_facecolor('linen')

    c_history = np.append(c_history, c_detail.iloc[1:].values, axis=0)
    d_history = np.append(d_history, d_detail.iloc[1:].values, axis=0)
    dd_history = np.append(dd_history, dd_detail.iloc[1:].values, axis=0)
    # return c_detail, d_detail, dd_detail
    ax2 = plt.subplot(gs[4:7, 3])
    ax2.set_title('Up-to-now Diffusion Summary', fontsize=title_size)
    draw_one_summary(c_history, d_history, dd_history, ax2)
    ax2.plot(0, 0, color='mediumturquoise', marker='o', markersize=16, linestyle='None', label='TC')
    ax2.plot(0, 0, color='lightgreen', marker='s', markersize=12, linestyle='None', label='TD')
    ax2.plot(0, 0, color='darkorange', marker='H', markersize=10, linestyle='None', label='TDD')
    legend = ax2.legend(loc='lower right', fontsize=14)
    legend.get_frame().set_facecolor('linen')
    exe_time += 1
    return c_history, d_history, dd_history


c_his, d_his, dd_his = np.empty((0, 3), int), np.empty((0, 3), int), np.empty((0, 3), int)

# c_his, d_his, dd_his = get_diffusion(c_his, d_his, dd_his)

# draw_whole_network()

window = tk.Tk()
window.title('my window')
window.geometry('200x100')

var = tk.StringVar()
l = tk.Label(window, text='Welcome!', textvariable=var, bg='green', font=('Arial', 12), width=15,
             height=2)
#l = tk.Label(window, text='OMG! this is TK!', bg='green', font=('Arial', 12), width=15, height=2)
l.pack()

on_hit = False
def hit_me():
    plt.close('all')

    global c_his
    global d_his
    global dd_his

    c_his, d_his, dd_his = get_diffusion(c_his, d_his, dd_his)


b = tk.Button(window, text='start diffusion', width=15, height=2, command=hit_me)
b.pack()

window.mainloop()
