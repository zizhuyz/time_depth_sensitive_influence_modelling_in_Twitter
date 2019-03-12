#!/usr/bin/env python
'all diffusion models'

import random
import networkx as nx
import pandas as pd
import time
import pre_cal_p_each_step as pre

print time.strftime('%X %x')


def TD_C(p1, time_frame, time_slot, N_steps, sds, n_iterations):
    edges = pd.read_csv('IN_structure.txt', sep='\t') # read the influence network
    # get users who have out-bound neighbours
    users = set(edges['u'].values)
    # generate the directed influence network
    g = nx.from_pandas_dataframe(edges, source='u', target='v', create_using=nx.DiGraph())
    # get the pre_cal_probabilities
    pre_ps = pre.load_pre_ps(p1, time_frame, time_slot)
    # get the u,v,prob
    p_df = pre.get_p_df(p1)
	# folder to save results
    TD_folder = "E:/TD/"
    for sd in sds:
        seed_users = []
        seed_users.append(sd)
        seed_active_set = set(seed_users)

        traces=[]
        traces.append([0, '', sd, 0])
        tracedf = '%sd_%s_%smin_%s_traces_detail_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        recordf = '%sd_%s_%smin_%s_records_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        cols1 = ['step', 'successful_Us', 'newly_activated_v', 'trial']
        results_df = pd.DataFrame(traces, columns=cols1)
        results_df.to_csv(tracedf, sep='\t', mode='a', header=cols1, index=False)
	
        record = []
        record.append([0, len(seed_active_set), 0])
        cols2 = ['step', 'n_newly_activated_v', 'trial']
        records_df = pd.DataFrame(record, columns=cols2)
        records_df.to_csv(recordf, sep='\t', mode='a', header=cols2, index=False)
        st = time.clock()
        for i in range(1, n_iterations):
            step = 0
            active_set = seed_active_set
            effect_set = seed_active_set
            # make a record of the activated_step of activated nodes
            temp = []
            temp.append([0, sd])
            temp_df=pd.DataFrame(temp, columns=['step', 'activated_node'])
            # life = []
            traces = []
            traces_detail = []
            record = []
            while step < N_steps:
                # Find the inactive neighbour set, which might be activated in this step
                neighbour_set = set()  # neighbour_set incremental? does not need to reset to empty
                new_active_set = set()
                for a_u in effect_set:
                    if a_u in users:  # if a_u has out-bound nodes
                        a_u_neighbour = g.successors(a_u)
                        neighbour_set = neighbour_set.union(a_u_neighbour)
                # Go through the inactive neighbour set, see which user will be activated based on probability
                neighbour_inactive_set = neighbour_set - active_set
                # users in neighbour set but not in active_set
                if len(neighbour_inactive_set) > 0:
                    step += 1
                    # Compute joint probability
                    for a_v in neighbour_inactive_set:
                        # Find this user's active followees
                        v_active_u_set = set(g.predecessors(a_v)) & effect_set  # intersection
                        # an active user u can always attempt to activated its vs within the timeframe
                        # if an active user u can only have one chance to activated its vs, change active_set to new_active_set
                        pi = 1.0
                        for u in v_active_u_set:
                            prob = p_df.loc[(p_df['u'] == u) & (p_df['v'] == a_v), 'prob'].iloc[0]
                            u_active_step = temp_df.loc[temp_df['activated_node'] == u, 'step'].iloc[0]
                            # p_each_step = get_decay_ps(prob, time_frame, time_slot)
                            p = pre_ps[prob][step-u_active_step-1] #decay of p step by step
                            pi *= (1.0 - p)
                        joint_prob = 1.0 - pi
                        x = random.random()
                        # If a neighbour is activated, add it into activated_set
                        if x < joint_prob:
                            new_active_set.add(a_v)
                            traces.append([step, v_active_u_set, a_v, i])
                            for u in v_active_u_set:
                                traces_detail.append([step, u, a_v, i])
                            temp_df=temp_df.append(pd.DataFrame([[step, a_v]], columns=['step', 'activated_node']),ignore_index=True)
                record.append([step, len(new_active_set), i])
                effect_set = set(temp_df[temp_df['step'] > step-int(time_frame/time_slot)]['activated_node'].tolist())
                active_set = active_set.union(new_active_set)
                if len(neighbour_inactive_set) == 0:
                    # life.append(step)
                    break
            # save the diffusion traces
            results = pd.DataFrame(traces_detail, columns=cols1)
            results.to_csv(tracedf, sep='\t', mode='a', header=False, index=False)
            # # save the statistics
            records = pd.DataFrame(record, columns=cols2)
            records.to_csv(recordf, sep='\t', mode='a', header=False, index=False)
            if i%50==0:
                print sd, time.clock()-st, 'seconds for', i, 'rounds of trials'


def TD_C_woi(p1, time_frame, time_slot, N_steps, sds, n_iterations):
    edges = pd.read_csv('IN_structure.txt', sep='\t')
    # get users who have out-bound neighbours
    users = set(edges['u'].values) 
    # generate the directed influence network
    g = nx.from_pandas_dataframe(edges, source='u', target='v', create_using=nx.DiGraph())
    # get the pre_cal_ps
    pre_ps = pre.load_pre_ps_woi(p1, time_frame, time_slot)
    # get the u,v,prob
    p_df = pre.get_p_df_woi(p1)

    TD_folder = "E:/TD/"

    for sd in sds:
        seed_users = []
        seed_users.append(sd)
        seed_active_set = set(seed_users)

        traces=[]
        traces.append([0,'', sd,0])
        tracedf = '%se_%s_%smin_%s_traces_detail_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        recordf = '%se_%s_%smin_%s_records_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        cols1 = ['step', 'successful_Us', 'newly_activated_v', 'trial']
        results_df.to_csv(tracedf, sep='\t', mode='a', header=cols1, index=False)

        record = []
        record.append([0, len(seed_active_set), 0])
        cols2 = ['step', 'n_newly_activated_v', 'trial']
        records_df = pd.DataFrame(record, columns=cols2)
        records_df.to_csv(recordf, sep='\t', mode='a', header=cols2, index=False)

        st = time.clock()
        for i in range(1, n_iterations):
            # st = time.clock()
            step = 0
            active_set = seed_active_set
            effect_set = seed_active_set
            # make a record of the activated_step of activated nodes
            temp = []
            temp.append([0, sd])
            temp_df=pd.DataFrame(temp, columns=['step', 'activated_node'])
            traces = []
            traces_detail = []
            record = []
            while step < N_steps:
                # Find the inactive neighbour set, which might be activated in this step
                neighbour_set = set()  # neighbour_set incremental? does not need to reset to empty
                new_active_set = set()
                for a_u in effect_set:
                    if a_u in users:  # if a_u has out-bound nodes
                        a_u_neighbour = g.successors(a_u)
                        neighbour_set = neighbour_set.union(a_u_neighbour)
                # Go through the inactive neighbour set, see which user will be activated based on probability
                neighbour_inactive_set = neighbour_set - active_set
                # users in neighbour set but not in active_set
                if len(neighbour_inactive_set) > 0:
                    step += 1
                    # Compute joint probability
                    for a_v in neighbour_inactive_set:
                        # Find this user's active followees
                        v_active_u_set = set(g.predecessors(a_v)) & effect_set  # intersection
                        # an active user u can always attempt to activated its vs within the timeframe
                        # if an active user u can only have one chance to activated its vs, change active_set to new_active_set
                        pi = 1.0
                        for u in v_active_u_set:
                            prob = p_df.loc[(p_df['u'] == u) & (p_df['v'] == a_v), 'prob'].iloc[0]
                            u_active_step = temp_df.loc[temp_df['activated_node'] == u, 'step'].iloc[0]
                            # p_each_step = get_decay_ps(prob, time_frame, time_slot)
                            p = pre_ps[prob][step-u_active_step-1] #decay of p step by step
                            pi *= (1.0 - p)
                        joint_prob = 1.0 - pi
                        x = random.random()
                        # If a neighbour is activated, add it into activated_set
                        if x < joint_prob:
                            new_active_set.add(a_v)
                            traces.append([step, v_active_u_set, a_v, i])
                            for u in v_active_u_set:
                                traces_detail.append([step, u, a_v, i])
                            temp_df=temp_df.append(pd.DataFrame([[step, a_v]], columns=['step', 'activated_node']),ignore_index=True)
                            # print '------', step, v_active_u_set, a_v
                record.append([step, len(new_active_set), i])
                effect_set = set(temp_df[temp_df['step'] > step-int(time_frame/time_slot)]['activated_node'].tolist())
                active_set = active_set.union(new_active_set)
                if len(neighbour_inactive_set) == 0:
                    # print i, "trial, no more users can be activated at step", step
                    break
            # save the diffusion traces
            results = pd.DataFrame(traces_detail, columns=cols1)
            results.to_csv(tracedf, sep='\t', mode='a', header=False, index=False)
            # # save the statistics
            records = pd.DataFrame(record, columns=cols2)
            records.to_csv(recordf, sep='\t', mode='a', header=False, index=False)
            if i % 50 == 0:
                print sd, time.clock()-st, 'seconds for', i, 'rounds of trials'


def TDD_C(p1, time_frame, time_slot, N_steps, sds,  depth_decay_factor, n_iterations):
    edges = pd.read_csv('IN_structure.txt', sep='\t')
    # get users who have out-bound neighbours
    users = set(edges['u'].values)
    # generate the directed influence network
    g = nx.from_pandas_dataframe(edges, source='u', target='v', create_using=nx.DiGraph())
    # get the pre_cal_ps
    pre_ps = pre.load_pre_ps(p1, time_frame, time_slot)
    # get the u,v,prob
    p_df = pre.get_p_df(p1)

    for sd in sds:
        seed_users = []
        seed_users.append(sd)
        seed_active_set = set(seed_users)
        TDD_folder = "E:/TDD/"
        traces=[]
        traces.append([0,'', sd,0])
        tracedf = '%sd2_%s_%smin_%s_traces_detail_1.txt'%(str(TDD_folder),str(sd),str(time_slot),str(p1))
        recordf = '%sd2_%s_%smin_%s_records_1.txt'%(str(TDD_folder),str(sd),str(time_slot),str(p1))
        cols1 = ['step', 'successful_Us', 'newly_activated_v', 'trial']
        results_df.to_csv(tracedf, sep='\t', mode='a', header=cols1, index=False)

        record = []
        record.append([0, len(seed_active_set), 0])
        cols2 = ['step', 'n_newly_activated_v', 'trial']
        records_df = pd.DataFrame(record, columns=cols2)
        records_df.to_csv(recordf, sep='\t', mode='a', header=cols2, index=False)
        st = time.clock()
        for i in range(1, n_iterations):
            step = 0
            active_set = seed_active_set
            effect_set = seed_active_set
            # make a record of the activated_step of activated nodes
            temp = []
            temp.append([0, sd])
            temp_df=pd.DataFrame(temp, columns=['step', 'activated_node'])
            traces = []
            traces_detail = []
            record = []
            # life = []
            while step < N_steps:
                # Find the inactive neighbour set, which might be activated in this step
                neighbour_set = set()  # neighbour_set incremental? does not need to reset to empty
                new_active_set = set()
                for a_u in effect_set:
                    if a_u in users:  # if a_u has out-bound nodes
                        a_u_neighbour = g.successors(a_u)
                        neighbour_set = neighbour_set.union(a_u_neighbour)
                # Go through the inactive neighbour set, see which user will be activated based on probability
                neighbour_inactive_set = neighbour_set - active_set
                # users in neighbour set but not in active_set
                if len(neighbour_inactive_set) > 0:
                    step += 1
                    # Compute joint probability
                    for a_v in neighbour_inactive_set:
                        # Find this user's active followees
                        v_active_u_set = set(g.predecessors(a_v)) & effect_set  # intersection
                        # an active user u can always attempt to activated its vs within the timeframe
                        # if an active user u can only have one chance to activated its vs, change active_set to new_active_set
                        pi = 1.0
                        for u in v_active_u_set:
                            prob = p_df.loc[(p_df['u'] == u) & (p_df['v'] == a_v), 'prob'].iloc[0]
                            u_active_step = temp_df.loc[temp_df['activated_node'] == u, 'step'].iloc[0]
                            # print 'step', step, 'user', u, 'u_active_step', u_active_step, 'index', step-u_active_step-1
                            spl_to_source = nx.shortest_path_length(g, source=sd, target=u)
							# apply decay factor
                            p = pre_ps[prob][step-u_active_step-1]*(depth_decay_factor**spl_to_source)
                            pi *= (1.0 - p)
                        joint_prob = 1.0 - pi
                        x = random.random()
                        # If a neighbour is activated, add it into activated_set
                        if x < joint_prob:
                            new_active_set.add(a_v)
                            traces.append([step, v_active_u_set, a_v, i])
                            for u in v_active_u_set:
                                traces_detail.append([step, u, a_v, i])
                            temp_df=temp_df.append(pd.DataFrame([[step, a_v]], columns=['step', 'activated_node']),ignore_index=True)
                effect_set = set(temp_df[temp_df['step'] > step-int(time_frame/time_slot)]['activated_node'].tolist())
                active_set = active_set.union(new_active_set)
                record.append([step, len(new_active_set), i])
                if len(neighbour_inactive_set) == 0:
                    # print i, "trial, no more users at step", step
                    # life.append(step)
                    break
            # save the diffusion trace details
            results = pd.DataFrame(traces_detail, columns=cols1)
            results.to_csv(tracedf, sep='\t', mode='a', header=False, index=False)
            # # save the statistics
            records = pd.DataFrame(record, columns=cols2)
            records.to_csv(recordf, sep='\t', mode='a', header=False, index=False)
            if i % 50 == 0:
                print sd, time.clock()-st, 'seconds for', i, 'rounds of trials'


def TDD_C_woi(p1, time_frame, time_slot, N_steps, sds, depth_decay_factor, n_iterations):
    edges = pd.read_csv('IN_structure.txt', sep='\t')
    # get users who have out-bound neighbours
    users = set(edges['u'].values) 
    # generate the directed influence network
    g = nx.from_pandas_dataframe(edges, source='u', target='v', create_using=nx.DiGraph())
    # get the pre_cal_ps
    pre_ps = pre.load_pre_ps_woi(p1, time_frame, time_slot)
    # get the u,v,prob
    p_df = pre.get_p_df_woi(p1)

    TDD_folder = "E:/TDD/"


    for sd in sds:
        seed_users = []
        seed_users.append(sd)
        seed_active_set = set(seed_users)
        traces=[]
        traces.append([0,'', sd,0])
        tracedf = '%se2_%s_%smin_%s_traces_detail_1.txt'%(str(TDD_folder),str(sd),str(time_slot),str(p1))
        recordf = '%se2_%s_%smin_%s_records_1.txt'%(str(TDD_folder),str(sd),str(time_slot),str(p1))
        cols1 = ['step', 'successful_Us', 'newly_activated_v', 'trial']
        results_df = pd.DataFrame(traces, columns=cols1)
        results_df.to_csv(tracedf, sep='\t', mode='a', header=cols1, index=False)

        record = []
        record.append([0, len(seed_active_set), 0])
        cols2 = ['step', 'n_newly_activated_v', 'trial']
        records_df = pd.DataFrame(record, columns=cols2)
        records_df.to_csv(recordf, sep='\t', mode='a', header=cols2, index=False)
        st = time.clock()
        for i in range(1, n_iterations):
            step = 0
            active_set = seed_active_set
            effect_set = seed_active_set
            # make a record of the activated_step of activated nodes
            temp = []
            temp.append([0, sd])
            temp_df=pd.DataFrame(temp, columns=['step', 'activated_node'])
            traces = []
            traces_detail = []
            record = []
            while step < N_steps:
                # Find the inactive neighbour set, which might be activated in this step
                neighbour_set = set()  # neighbour_set incremental? does not need to reset to empty
                new_active_set = set()

                for a_u in effect_set:
                    if a_u in users:  # if a_u has out-bound nodes
                        a_u_neighbour = g.successors(a_u)
                        neighbour_set = neighbour_set.union(a_u_neighbour)
                # Go through the inactive neighbour set, see which user will be activated based on probability
                neighbour_inactive_set = neighbour_set - active_set
                # users in neighbour set but not in active_set
                if len(neighbour_inactive_set) > 0:
                    step += 1
                    # Compute joint probability
                    for a_v in neighbour_inactive_set:
                        # Find this user's active followees
                        v_active_u_set = set(g.predecessors(a_v)) & effect_set  # intersection
                        # an active user u can always attempt to activated its vs within the timeframe
                        # if an active user u can only have one chance to activated its vs, change active_set to new_active_set
                        pi = 1.0
                        for u in v_active_u_set:
                            prob = p_df.loc[(p_df['u'] == u) & (p_df['v'] == a_v), 'prob'].iloc[0]
                            u_active_step = temp_df.loc[temp_df['activated_node'] == u, 'step'].iloc[0]
                            # p_each_step = get_decay_ps(prob, time_frame, time_slot)
                            spl_to_source = nx.shortest_path_length(g,source=sd,target=u)
							# apply decay factor
                            p = pre_ps[prob][step-u_active_step-1]*(depth_decay_factor**spl_to_source)
                            pi *= (1.0 - p)
                        joint_prob = 1.0 - pi
                        x = random.random()
                        # If a neighbour is activated, add it into activated_set
                        if x < joint_prob:
                            # print 'joint_prob:', joint_prob, 'random number:', x
                            new_active_set.add(a_v)
                            traces.append([step, v_active_u_set, a_v, i])
                            for u in v_active_u_set:
                                traces_detail.append([step, u, a_v, i])
                            temp_df=temp_df.append(pd.DataFrame([[step, a_v]], columns=['step', 'activated_node']),ignore_index=True)
                record.append([step, len(new_active_set), i])
                effect_set = set(temp_df[temp_df['step'] > step-int(time_frame/time_slot)]['activated_node'].tolist())
                active_set = active_set.union(new_active_set)
                if len(neighbour_inactive_set) == 0:
                    break
            # save the diffusion traces
            results = pd.DataFrame(traces_detail, columns=cols1)
            results.to_csv(tracedf, sep='\t', mode='a', header=False, index=False)
            # save the statistics
            records = pd.DataFrame(record, columns=cols2)
            records.to_csv(record, sep='\t', mode='a', header=False, index=False)
            if i % 50 == 0:
                print sd, time.clock()-st, 'seconds for', i, 'rounds of trials'


def TC_C(p1, time_frame, time_slot, N_steps, sds, n_iterations):
    edges = pd.read_csv('IN_structure.txt', sep='\t')
    # get users who have out-bound neighbours
    users = set(edges['u'].values)
    # generate the directed influence network
    g = nx.from_pandas_dataframe(edges, source='u', target='v', create_using=nx.DiGraph())
    # get the pre_cal_ps
    pre_ps = pre.load_pre_c_ps(p1, time_frame, time_slot)
    # get the u,v,prob
    p_df = pre.get_p_df(p1)

    for sd in sds:
        seed_users = []
        seed_users.append(sd)
        seed_active_set = set(seed_users)
        TD_folder = "E:/TC/"
        traces=[]
        traces.append([0, '', sd, 0])
        tracedf = '%sd_cp_%s_%smin_%s_traces_detail_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        recordf = '%sd_cp_%s_%smin_%s_records_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        cols1 = ['step', 'successful_Us', 'newly_activated_v', 'trial']
        results_df = pd.DataFrame(traces, columns=cols1)
        results_df.to_csv(tracedf, sep='\t', mode='a', header=cols1, index=False)

        record = []
        record.append([0, len(seed_active_set), 0])
        cols2 = ['step', 'n_newly_activated_v', 'trial']
        records_df = pd.DataFrame(record, columns=cols2)
        records_df.to_csv(recordf, sep='\t', mode='a', header=cols2, index=False)
        st=time.clock()
        for i in range(1, n_iterations):
            step = 0
            active_set = seed_active_set
            effect_set = seed_active_set
            # make a record of the activated_step of activated nodes
            temp = []
            # for seed in seed_active_set:
            #     temp.append([0, seed])
            temp.append([0, sd])
            temp_df=pd.DataFrame(temp, columns=['step', 'activated_node'])
            traces = []
            traces_detail = []
            record = []
            # life = []
            while step < N_steps:
                # Find the inactive neighbour set, which might be activated in this step
                neighbour_set = set()  # neighbour_set incremental? does not need to reset to empty
                new_active_set = set()
                for a_u in effect_set:
                    if a_u in users:  # if a_u has out-bound nodes
                        a_u_neighbour = g.successors(a_u)
                        neighbour_set = neighbour_set.union(a_u_neighbour)
                # Go through the inactive neighbour set, see which user will be activated based on probability
                neighbour_inactive_set = neighbour_set - active_set
                # users in neighbour set but not in active_set
                if len(neighbour_inactive_set) > 0:
                    step += 1
                    # Compute joint probability
                    for a_v in neighbour_inactive_set:
                        # Find this user's active followees
                        v_active_u_set = set(g.predecessors(a_v)) & effect_set  # intersection
                        # an active user u can always attempt to activated its vs within the timeframe
                        # if an active user u can only have one chance to activated its vs, change active_set to new_active_set
                        pi = 1.0
                        for u in v_active_u_set:
                            prob = p_df.loc[(p_df['u'] == u) & (p_df['v'] == a_v), 'prob'].iloc[0]
                            # u_active_step = temp_df.loc[temp_df['activated_node'] == u, 'step'].iloc[0]
                            p = pre_ps[prob] # constant p in each step by step
                            pi *= (1.0 - p)
                        joint_prob = 1.0 - pi
                        x = random.random()
                        # If a neighbour is activated, add it into activated_set
                        if x < joint_prob:
                            new_active_set.add(a_v)
                            traces.append([step, v_active_u_set, a_v, i])
                            for u in v_active_u_set:
                                traces_detail.append([step, u, a_v, i])
                            temp_df=temp_df.append(pd.DataFrame([[step, a_v]], columns=['step', 'activated_node']),ignore_index=True)
                effect_set = set(temp_df[temp_df['step'] > step-int(time_frame/time_slot)]['activated_node'].tolist())
                active_set = active_set.union(new_active_set)
                record.append([step, len(new_active_set), i])
                if len(neighbour_inactive_set) == 0:
                    # life.append(step)
                    break
            # save the diffusion traces
            results = pd.DataFrame(traces_detail, columns=cols1)
            results.to_csv(tracedf, sep='\t', mode='a', header=False, index=False)
            # # save the statistics
            records = pd.DataFrame(record, columns=cols2)
            records.to_csv(recordf, sep='\t', mode='a', header=False, index=False)
            if i % 50 == 0:
                print sd, time.clock()-st, 'seconds for', i, 'rounds of trials'


def TC_C_woi(p1, time_frame, time_slot, N_steps, sds, n_iterations):
    edges = pd.read_csv('IN_structure.txt', sep='\t')
    # get users who have out-bound neighbours
    users = set(edges['u'].values)  # 2946 users who have out-bound edges
    # generate the directed influence network
    g = nx.from_pandas_dataframe(edges, source='u', target='v', create_using=nx.DiGraph())
    # get the pre_cal_ps
    pre_ps = pre.load_pre_c_ps_woi(p1, time_frame, time_slot)
    # get the u,v,prob
    p_df = pre.get_p_df_woi(p1)

    TD_folder = "E:/TC/"

    for sd in sds:
        seed_users = []
        seed_users.append(sd)
        seed_active_set = set(seed_users)
        traces=[]
        traces.append([0, '', sd, 0])
        tracedf = '%se_cp_%s_%smin_%s_traces_detail_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        recordf = '%se_cp_%s_%smin_%s_records_1.txt'%(str(TD_folder),str(sd),str(time_slot),str(p1))
        cols1 = ['step', 'successful_Us', 'newly_activated_v', 'trial']
        results_df = pd.DataFrame(traces, columns=cols1)
        results_df.to_csv(tracedf, sep='\t', mode='a', header=cols1, index=False)

        record = []
        record.append([0, len(seed_active_set), 0])
        cols2 = ['step', 'n_newly_activated_v', 'trial']
        records_df = pd.DataFrame(record, columns=cols2)
        records_df.to_csv(recordf, sep='\t', mode='a', header=cols2, index=False)
        st=time.clock()
        for i in range(1, n_iterations):
            step = 0
            active_set = seed_active_set
            effect_set = seed_active_set
            # make a record of the activated_step of activated nodes
            temp = []
            temp.append([0, sd])
            temp_df=pd.DataFrame(temp, columns=['step', 'activated_node'])
            traces = []
            traces_detail = []
            record = []
            while step < N_steps:
                # Find the inactive neighbour set, which might be activated in this step
                neighbour_set = set()  # neighbour_set incremental? does not need to reset to empty
                new_active_set = set()
                for a_u in effect_set:
                    if a_u in users:  # if a_u has out-bound nodes
                        a_u_neighbour = g.successors(a_u)
                        neighbour_set = neighbour_set.union(a_u_neighbour)
                # Go through the inactive neighbour set, see which user will be activated based on probability
                neighbour_inactive_set = neighbour_set - active_set
                # users in neighbour set but not in active_set
                if len(neighbour_inactive_set) > 0:
                    step += 1
                    # Compute joint probability
                    for a_v in neighbour_inactive_set:
                        # Find this user's active followees
                        v_active_u_set = set(g.predecessors(a_v)) & effect_set  # intersection
                        # an active user u can always attempt to activated its vs within the timeframe
                        # if an active user u can only have one chance to activated its vs, change active_set to new_active_set
                        pi = 1.0
                        for u in v_active_u_set:
                            prob = p_df.loc[(p_df['u'] == u) & (p_df['v'] == a_v), 'prob'].iloc[0]
                            # u_active_step = temp_df.loc[temp_df['activated_node'] == u, 'step'].iloc[0]
                            p = pre_ps[prob] #constant p in each step by step
                            pi *= (1.0 - p)
                        joint_prob = 1.0 - pi
                        x = random.random()
                        # If a neighbour is activated, add it into activated_set
                        if x < joint_prob:
                            new_active_set.add(a_v)
                            traces.append([step, v_active_u_set, a_v, i])
                            for u in v_active_u_set:
                                traces_detail.append([step, u, a_v, i])
                            temp_df=temp_df.append(pd.DataFrame([[step, a_v]], columns=['step', 'activated_node']),ignore_index=True)
                effect_set = set(temp_df[temp_df['step'] > step-int(time_frame/time_slot)]['activated_node'].tolist())
                active_set = active_set.union(new_active_set)
                record.append([step, len(new_active_set), i])
                if len(neighbour_inactive_set) == 0:
                    break
            # save the diffusion traces
            results = pd.DataFrame(traces_detail, columns=cols1)
            results.to_csv(tracedf, sep='\t', mode='a', header=False, index=False)
            # # save the statistics
            records = pd.DataFrame(record, columns=cols2)
            records.to_csv(recordf, sep='\t', mode='a', header=False, index=False)
            if i % 50 == 0:
                print sd, time.clock()-st, 'seconds for', i, 'rounds of trials'
