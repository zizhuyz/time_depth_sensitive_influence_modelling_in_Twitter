#!/usr/bin/env python
'pre-calculate the decaying Ps in each step for all the possible P values'
import pandas as pd
import numpy as np
from scipy import integrate
import pickle
from sympy import solve, Symbol,log

# p1: It is assumed that there is always a small influence probability between connected user pairs in the influence network, which is denoted as $p_1$.

# calculate influence prob. with explicit and implicit interactions
def get_p_df(p1):
    # read the influence network strucutre in
    edges = pd.read_csv('IN_structure.txt', sep='\t')
    # assign probs for user pairs with no interaction history
    temp = edges.loc[edges['edge_type'].isin([1,2])]
    columns = ['u', 'v']
    p1_df = pd.DataFrame(temp, columns=columns).copy()
    p1_df['prob'] = p1
    # print p1_df.shape  # 56719 (u,v) directed edges

    # read the computed probs data
    interactions = pd.read_csv('..\\04 compute probs\\r2.txt', sep='\t')
    # print p2_df  # 2080 (u,v) directed edges
    interactions['prob'] = interactions.apply(lambda row: 1-(1-p1)*(1-row.r2), axis=1)
    # p2_df['prob'] = 1-(1-p1)*(1-p2_df['p2'])
    columns = ['u', 'v', 'prob']
    p2_df = pd.DataFrame(interactions, columns=columns).copy()
    # put two dataframes together (58799*3)
    # p_df is the u, v probability dataframe to be used in the IC model
    p_df = p1_df.append(p2_df, ignore_index=True)
    return p_df

# calculate influence prob. with explicit interactions only
def get_p_df_woi(p1):
    # read the influence network strucutre in
    edges = pd.read_csv('IN_structure.txt', sep='\t')

    # assign probs for user pairs with no interaction history
    temp = edges.loc[edges['edge_type'].isin([1,2])]
    columns = ['u', 'v']
    p1_df = pd.DataFrame(temp, columns=columns).copy()
    p1_df['prob'] = p1
    # print p1_df.shape  # 56719 (u,v) directed edges

    # read the computed probs data
    interactions = pd.read_csv('..\\04 compute probs\\r2_woi.txt', sep='\t')
    # print p2_df  # 2080 (u,v) directed edges
    implicit = interactions.loc[interactions['class'].isin([3])].copy()
    # class 3 contains the (u,v) pairs that only have implicit interactions
    implicit['prob'] = p1
    # print implicit #[451 rows x 5 columns]
    p1_df = p1_df.append(implicit[['u', 'v', 'prob']])  # [57170 rows x 3 columns]

    explicit = interactions.loc[interactions['class'].isin([1, 2])].copy()
    # class 2 are the (u,v) pairs that have both implicit and explicit interactions
    # class 1 are the (u,v) pairs that only have explicit interactions
    explicit['prob'] = explicit.apply(lambda row: 1-(1-p1)*(1-row.r2_woi), axis=1)
    p2_df = explicit[['u', 'v', 'prob']]  # [1629 rows x 5 columns]

    # put two dataframes together (58799*3)
    # p_df is the u, v probability dataframe to be used in the IC model
    p_df = p1_df.append(p2_df, ignore_index=True)
    # print p_df  # 58799 (u, v) directed edges
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


def get_pre_cal_ps(p1, time_frame, time_slot):
    pre_cal_ps = {}
    p_df = get_p_df(p1)
    # print p_df.shape
    p = p_df['prob'].unique()
    # print len(p)
    for a_p in np.nditer(p):
        ps_of_a_p = get_decay_ps(a_p, time_frame, time_slot)
        # print ps_of_a_p
        pre_cal_ps[np.asscalar(a_p)] = ps_of_a_p
    return pre_cal_ps

def get_pre_cal_ps_smaller(p1, time_frame, time_slot, alpha):
    pre_cal_ps = {}
    p_df = get_p_df(p1)
    print p_df.nlargest(10, 'prob')
    p_df['prob'] = p_df['prob'].apply(lambda x: x*alpha)
    print p_df.nlargest(10, 'prob')
    # print p_df.shape
    p = p_df['prob'].unique()
    # print len(p)
    for a_p in np.nditer(p):
        ps_of_a_p = get_decay_ps(a_p, time_frame, time_slot)
        # print ps_of_a_p
        pre_cal_ps[np.asscalar(a_p)] = ps_of_a_p
    return pre_cal_ps

def get_pre_cal_ps_woi(p1, time_frame, time_slot):
    pre_cal_ps = {}
    p_df = get_p_df_woi(p1)
    p = p_df['prob'].unique()
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

def get_pre_cal_c_ps(p1, time_frame, time_slot):
    pre_cal_c_ps = {}
    p_df = get_p_df(p1)
    p = p_df['prob'].unique()
    for a_p in np.nditer(p):
        p_step_of_a_p = get_constant_ps(a_p, time_frame, time_slot)
        # print ps_of_a_p
        pre_cal_c_ps[np.asscalar(a_p)] = p_step_of_a_p
    return pre_cal_c_ps

def get_pre_cal_c_ps_woi(p1, time_frame, time_slot):
    pre_cal_c_ps_woi = {}
    p_df = get_p_df_woi(p1)
    p = p_df['prob'].unique()
    for a_p in np.nditer(p):
        p_step_of_a_p = get_constant_ps(a_p, time_frame, time_slot)
        pre_cal_c_ps_woi[np.asscalar(a_p)] = p_step_of_a_p
    return pre_cal_c_ps_woi

def load_pre_c_ps(p1, time_frame,time_slot ):
    with open('PreP\pre_c_ps'+str(p1)+'_'+str(time_frame)+'_'+str(time_slot)+'.pkl', 'rb') as f:
        return pickle.load(f)

def load_pre_c_ps_woi(p1, time_frame,time_slot ):
    with open('PreP\pre_c_ps_woi'+str(p1)+'_'+str(time_frame)+'_'+str(time_slot)+'.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pre_ps(p1, time_frame,time_slot ):
    with open('PreP\pre_ps_'+str(p1)+'_'+str(time_frame)+'_'+str(time_slot)+'.pkl', 'rb') as f:
        return pickle.load(f)

def load_pre_ps_smaller(p1, time_frame,time_slot, alpha):
    with open('PreP\pre_ps_'+str(p1)+'_'+str(alpha)+'_'+str(time_frame)+'_'+str(time_slot)+'.pkl', 'rb') as f:
        return pickle.load(f)

def load_pre_ps_woi(p1, time_frame,time_slot ):
    with open('PreP\pre_ps_woi'+str(p1)+'_'+str(time_frame)+'_'+str(time_slot)+'.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    p = [0]
    TIME_FRAME = 300
    slots = [15]
    al = 0.8
    for p1 in p:
        for slot in slots:
            ps = get_pre_cal_ps(p1, TIME_FRAME, slot)
            save_obj(ps, 'pre_ps_'+str(p1)+'_'+str(TIME_FRAME)+'_'+str(slot))
            # ps_woi = get_pre_cal_ps_woi(p1,TIME_FRAME,slot)
            # save_obj(ps_woi,'pre_ps_woi'+str(p1)+'_'+str(TIME_FRAME)+'_'+str(slot))
            # c_ps = get_pre_cal_c_ps(p1, TIME_FRAME,slot)
            # save_obj(c_ps,'pre_c_ps'+str(p1)+'_'+str(TIME_FRAME)+'_'+str(slot))
            # c_ps_woi = get_pre_cal_c_ps_woi(p1,TIME_FRAME,slot)
            # save_obj(c_ps_woi,'pre_c_ps_woi'+str(p1)+'_'+str(TIME_FRAME)+'_'+str(slot))
            print 'one calculation finished', p1, TIME_FRAME, slot

