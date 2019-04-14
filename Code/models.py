#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/03/2019 5:42 PM
# @Author  : Zizhu
# @File    : models.py
import random
import networkx as nx
import pandas as pd
import time
import cal_prob as cal
import numpy as np


def get_inputs_for_d_model(history, sn, time_frame, time_slot, p1=0.001, alpha=1, beta=0.5):
    """
    :param history: interaction history
    :param sn: social network
    :param time_frame:
    :param time_slot:
    :param p1:
    :param alpha:
    :param beta:
    :return: p_df (u,v,prob), G (network), pre_ps (of each steps)
    """
    # get the u,v,prob
    p_df = cal.get_p_df(history, sn, p1=p1, beta=beta)
    # the influence network
    G = nx.from_pandas_edgelist(p_df, source='u', target='v', edge_attr='prob', create_using=nx.DiGraph())
    # the p in each time slot
    pre_ps = cal.get_pre_cal_decay_ps(time_frame, time_slot, p_df, alpha)
    return p_df, pre_ps, G


def get_inputs_for_c_model(history, sn, time_frame, time_slot, p1=0.001, alpha=1, beta=0.5):
    """
    :param history: interaction history
    :param sn: social network
    :param time_frame:
    :param time_slot:
    :param p1:
    :param alpha:
    :param beta:
    :return: p_df (u,v,prob), G (network), pre_ps (of each steps)
    """
    # get the u,v,prob
    p_df = cal.get_p_df(history, sn, p1=p1, beta=beta)
    # the influence network
    G = nx.from_pandas_edgelist(p_df, source='u', target='v', edge_attr='prob', create_using=nx.DiGraph())
    # the p in each time slot
    pre_ps = cal.get_pre_cal_const_ps(time_frame, time_slot, p_df, alpha)
    return p_df, pre_ps, G


def model(md, pre_ps, G, time_frame, time_slot, N_steps, sd, alpha=1, depth_decay_factor=0.2):
    """
    :param md: name of diffusion model, must be "TD", "TDD" or "TC"
    :param p_df: probs for md, adjusted by "beta"(implicit influence weight)
    :param pre_ps: probs at each step for md, adjusted by "alpha"
    :param G: network structure
    :param time_frame: T0, after T0, a user's influence drops to 0, 300mins
    :param time_slot: ts, how long is each step
    :param N_steps: how many diffusion steps
    :param sd: seed user
    :param depth_decay_factor: in TDD model
    :return:
    """
    # parameter check
    if md not in ['TD', 'TDD', 'td', 'tdd', 'TC', 'tc']:
        print('No such model! Model should be TD, TDD or TC')
        exit()

    # all users
    users = set(G.nodes)
    # p_df 'prob' scaled by alpha

    # seed
    seed_active_set = set(sd)

    # shortest path length to seed
    depth_dict = nx.single_source_shortest_path_length(G, sd)

    st = time.clock()

    step = 0
    active_set = seed_active_set
    effect_set = seed_active_set
    # make a record of the activated_step of activated nodes
    trace = []
    trace.append([0, seed_active_set, {}])

    temp = []
    temp.append([0, sd, 0])
    temp_df = pd.DataFrame(temp, columns=['step', 'activated_node', 'depth'])

    n_lose_effect=0

    while step < N_steps:
        # Find the inactive neighbour set, which might be activated in this step
        neighbour_set = set()
        new_active_set = set()
        for a_u in effect_set:
            if a_u in users:  # if a_u has out-bound nodes
                a_u_neighbour = G.successors(a_u)
                neighbour_set = neighbour_set.union(a_u_neighbour)
        # Go through the inactive neighbour set, see which user will be activated based on probability
        neighbour_inactive_set = neighbour_set - active_set
        # users in neighbour set but not in active_set
        if len(neighbour_inactive_set) > 0:
            step += 1
            # Compute joint probability
            for a_v in neighbour_inactive_set:
                # Find this user's active followees
                v_active_u_set = set(G.predecessors(a_v)) & effect_set  # intersection
                # an active user u can always attempt to activated its vs within the timeframe
                # if an active user u can only have one chance to activated its vs, change active_set to new_active_set
                pi = 1.0
                for u in v_active_u_set:
                    # prob = p_df.loc[(p_df['u'] == u) & (p_df['v'] == a_v), 'prob'].iloc[0]
                    prob = G[u][a_v]['prob']*alpha
                    u_active_step = temp_df.loc[temp_df['activated_node'] == u, 'step'].iloc[0]
                    if md in ['TD', 'td']:
                        p = pre_ps[prob][step-u_active_step-1] # decay of p step by step
                    if md in['TDD', 'tdd']:
                        p = pre_ps[prob][step - u_active_step - 1]
                        spl_to_source = nx.shortest_path_length(G, source=sd, target=u)
                        p = p*(depth_decay_factor**spl_to_source)
                    if md in ['TC', 'tc']:
                        p = pre_ps[prob]
                    pi *= (1.0 - p)
                joint_prob = 1.0 - pi
                x = random.random()/int(time_frame/time_slot)*3
                # If a neighbour is activated, add it into activated_set
                if x < joint_prob:
                    # print 'joint_prob:', joint_prob, 'random number:', x
                    new_active_set.add(a_v)
                    a_v_depth_to_sd = depth_dict[a_v]
                    temp_df=temp_df.append(pd.DataFrame([[step, a_v, a_v_depth_to_sd]], columns=['step', 'activated_node', 'depth']),ignore_index=True)
        effect_set = set(temp_df[temp_df['step'] > step-int(time_frame/time_slot)]['activated_node'].tolist())
        active_set = active_set.union(new_active_set)
        lose_effect_nd0 = active_set-effect_set
        if len(new_active_set):
            trace.append([step, effect_set, lose_effect_nd0])
        if len(lose_effect_nd0)>n_lose_effect:
            n_lose_effect += 1
            trace.append([step, effect_set, lose_effect_nd0])

        if len(neighbour_inactive_set) == 0:
            break
    trace_df = pd.DataFrame(trace, columns=['step', 'effect_nd1', 'lose_effect_nd0'])
    print sd, time.clock() - st, 'seconds for one simulation'
    print trace_df
    print temp_df
    return trace_df, temp_df


if __name__ == "__main__":
    interactions = pd.read_csv('factors1.txt', sep=',')
    network = pd.read_csv('network1.txt', sep=',')
    p1, alphaTC, alphaTD, beta = 0.001, 1.5, 2, 0.5
    time_frame = 300
    time_slot = 10
    # p_df, pre_ps, G = get_inputs_for_d_model(interactions, network, time_frame=time_frame, time_slot=time_slot,alpha=alphaTD)
    # rslt_plot, rstl_stat = model('TD', pre_ps, G, time_frame, time_slot, 50, 'j', alpha=alphaTD)
    c_p_df, c_pre_ps, c_G = get_inputs_for_c_model(interactions, network, time_frame=time_frame, time_slot=time_slot, alpha=alphaTC)
    rslt = model('TC', c_pre_ps, c_G, time_frame, time_slot, 50, 'j', alpha=alphaTC)

