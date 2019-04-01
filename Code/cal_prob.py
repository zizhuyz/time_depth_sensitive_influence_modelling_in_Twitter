#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/03/2019 8:46 AM
# @Author  : Zizhu
# @File    : DemoRun.py
import pandas as pd
import numpy as np
from scipy import integrate
from sympy import solve, Symbol,log
import pickle


def adjust(n_actions):
    return 1/(1+np.exp(1-n_actions))


def jointprob(p_1, p_2):
    return 1-(1-p_1)*(1-p_2)


def get_p_df(history, sn, p1=0.001, beta=0.5):
    """
    :param p1: the small prob between any (u,v) pairs
    :param interactions: interaction history
    :param sn: influence network structure
    :param beta: the weight of implicit influence, should be in (0,1], default = 0.5
    :return: influence probs of each (u,v) pair in sn, DataFrame ['u','v','prob']
    """
    # with implicit influence
    data_0 = history[history['A_u_star']==0].copy()  # if A_u_star is 0, it cannot go to the fraction
    data_0['p_e'] = 0
    data_0['p_i'] = adjust(data_0['A_u'])*beta*(data_0['A_u2v_i']/data_0['A_u'])
    data_0['p_2'] = jointprob(data_0['p_e'], data_0['p_i'])

    data_1 = history[history['A_u_star']>0].copy()
    data_1['p_e'] = adjust(data_1['A_u_star'])*data_1['A_u2v_e']/data_1['A_u_star']
    data_1['p_i'] = adjust(data_1['A_u'])*beta*(data_1['A_u2v_i']/data_1['A_u'])
    data_1['p_2'] = jointprob(data_1['p_e'], data_1['p_i'])

    p_2 = data_0.append(data_1, ignore_index=True)

    # assign probs for user pairs with no interaction history
    # edge type: 1: follow only, u has no actions 2: follow only, u has actions
    # 3, (u,v) pairs with interaction and following relations
    # 4, (u,v) pairs only with interaction relations
    temp = sn.loc[sn['edge_type'].isin([1, 2])]
    columns = ['u', 'v']
    p1_df = pd.DataFrame(temp, columns=columns).copy()
    p1_df['prob'] = p1

    p_2['prob'] = p_2.apply(lambda row: 1 - (1 - p1) * (1 - row.p_2), axis=1)
    columns = ['u', 'v', 'prob']
    p2_df = pd.DataFrame(p_2, columns=columns).copy()
    # put two dataframes together, p_df is the u, v probability dataframe to be used in the IC model
    p_df = p1_df.append(p2_df, ignore_index=True)
    return p_df


def get_density(time_frame, time_slot):
    k, l , A  = np.array([0.63328779, 85.57088989, 1.0423])
    my_list = []
    for n in np.arange(0,time_frame,time_slot):
        val, err= integrate.quad(lambda x: A*float(k)/l*(x/float(l))**(k-1)*np.exp(-(x/float(l))**k),n, n+time_slot)
        # print n, n+time_slot, val
        my_list.append(float(format(val,'.6f')))
    return my_list


def get_decay_ps(t_p, time_frame, time_slot):
    x = Symbol('x')  # p1 in the first time slot
    # get the coefficients of the p2, p3, pi... against p1, decaying obeys the densities
    dlist = get_density(time_frame=time_frame,time_slot=time_slot)
    co = np.array(dlist,dtype='f')/dlist[0]
    myfunc = np.log(1-t_p)+np.sum(co)*x  # approximate of 1-(1-p1)(1-p2)..(1-pi) = p based on ln(1-x)~-x when x is a small positive number
    p1 = solve(myfunc, x)
    # print p1
    ps = co*p1[0]
    return ps


def get_pre_cal_decay_ps(time_frame, time_slot, p_df, alpha=1):
    """
    :param p1: samll prob
    :param time_frame: T0, a user's influence drops to 0 after T0 time, e.g. 300mins
    :param time_slot: model proceed every time_slot, e.g. 5mins
    :param p_df: 'u' 'v' 'prob'
    :param alpha: adjust p1, (0,inf)
    :return: pre_cal_decay_ps, a dict {'0.001':{0.0006, 0.00002,....}, '0.2':{0.12, 0.09,...}}
    """
    pre_cal_ps = {}
    # you need to do a deep copy of the p_df
    pdf = p_df.copy()
    pdf['prob'] = pdf['prob'].apply(lambda x: x * alpha)
    p = pdf['prob'].unique()
    # print len(p)
    for a_p in np.nditer(p):
        ps_of_a_p = get_decay_ps(a_p, time_frame, time_slot)
        # print ps_of_a_p
        pre_cal_ps[np.asscalar(a_p)] = ps_of_a_p
    return pre_cal_ps


def get_constant_ps(t_p, time_frame, time_slot):
    N = time_frame/time_slot
    p_step = 1-np.power(1-t_p, 1/float(N))
    return p_step


def get_pre_cal_const_ps(time_frame, time_slot, p_df, alpha=1):
    """
    :param time_frame:
    :param time_slot:
    :param p_df:
    :param alpha: controls the cascade size. need to put alpha to p_df first and get the ps in each step
    :return:
    """
    pre_cal_c_ps = {}
    # p_df['prob'] = p_df['prob'].apply(lambda x: x * alpha)   # will update the probs in p_df
    # you need to do a deep copy of the p_df
    pdf = p_df.copy()
    pdf['prob'] = pdf['prob'].apply(lambda x: x * alpha)
    # the unique probs in influence probabilities
    p = pdf['prob'].unique()
    for a_p in np.nditer(p):
        p_step_of_a_p = get_constant_ps(a_p, time_frame, time_slot)
        # print ps_of_a_p
        pre_cal_c_ps[np.asscalar(a_p)] = p_step_of_a_p
    return pre_cal_c_ps


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    interactions = pd.read_csv('factors1.txt', sep=',')
    network = pd.read_csv('network1.txt', sep=',')
    p1, alpha1, alpha2, beta = 0.001, 1, 1, 0.5
    TIME_FRAME = 300
    slots = [5, 10, 15]
    p_df = get_p_df(interactions, network, p1=p1, beta=beta)
    # for slot in slots:
    #     ps = get_pre_cal_decay_ps(TIME_FRAME, slot, p_df, alpha=alpha1)
    #     save_obj(ps, 'pre_ps_' + str(p1) + '_' + str(alpha) + '_' + str(TIME_FRAME) + '_' + str(slot))
    #
    #     ps = get_pre_cal_const_ps(TIME_FRAME, slot, p_df, alpha=alpha2)
    #     save_obj(ps, 'pre_c_ps_' + str(p1) + '_' + str(alpha) + '_' + str(TIME_FRAME) + '_' + str(slot))