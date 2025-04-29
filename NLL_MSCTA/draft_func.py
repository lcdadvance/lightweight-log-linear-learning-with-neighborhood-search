# -*- coding: utf-8 -*-
"""

@author: am is are
"""

import numpy
import random
import copy
import time

#Get player set
def get_players(C):
    satellite_players=[]
    for i in range(len(C)): 
        player_info={}
        player_info['id']=i+1
        player_info['neighbor_set']=[]
        feasible_tasks=[]
        for j in range(len(C[0])):
            #if C[i][j]<float('inf'):
            if C[i][j]<10**10:
                feasible_tasks.append(j+1)
        player_info['feasible_tasks']=feasible_tasks
        subsets_fea_tasks=subsets(feasible_tasks)
        player_info['action_set']=subsets_fea_tasks
        
        satellite_players.append(player_info)
    return satellite_players

#Find all subsets of a set
def subsets(set_input):
    res=[[]]
    for element in set_input:
        res+=[i + [element] for i in res]
    return res

#Define neighbors
def get_neighbors(players_set): 
    for i in range(len(players_set)-1):
        for m in range(i+1,len(players_set)):
            if set(players_set[i]['feasible_tasks']).isdisjoint(set(players_set[m]['feasible_tasks']))==False:
                players_set[i]['neighbor_set'].append(players_set[m]['id'])
                players_set[m]['neighbor_set'].append(players_set[i]['id'])
    return players_set

#Initialize actions and memory vector
def intial_(players_set,memory_length):
    for x in players_set:
        x['action']=random.choice(x['action_set'])
        memory_vector=[]
        for i in range(memory_length):
            memory_vector.append(random.choice(x['action_set']))
        x['memory_vector']=memory_vector
    return players_set

#Task execution costs
def cal_overall_cost(C,D,id1,a):
    if len(a)==0:
        cost=0
    else:
        cost=C[id1-1][a[0]-1]+10*abs((D[id1-1][a[0]-1]-0))
        if len(a)>1:
            for i in range(1,len(a)):
                cost+=C[id1-1][a[i]-1]+10*abs((D[id1-1][a[i]-1]-D[id1-1][a[i-1]-1]))
    return cost

#global objective
def cal_global_obj_func(C,D,players_set):
    undertaken_tasks=[]
    for x in players_set:
        if len(x['action'])>0:
            for j in x['action']:
                undertaken_tasks.append(j)
    undertaken_tasks=set(undertaken_tasks) 
    left_tasks=set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-undertaken_tasks 
    global_obj_func = 10000*len(left_tasks)
    for y in players_set:
        global_obj_func+=cal_overall_cost(C, D, y['id'], y['action'])
    return global_obj_func
        
#individual utility 1     
def cal_local_uti(C,D,id1,fea_tasks,neighbors,a,players_set):
    executing_tasks = a[:]
    for s in neighbors: 
        if len(players_set[s-1]['action'])>0:
            for neig_a in players_set[s-1]['action']:
                executing_tasks.append(neig_a)
    executing_tasks=set(executing_tasks) 
    remain_tasks=set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-executing_tasks
    unassign_but_fea=set(fea_tasks) & remain_tasks
    local_uti= -cal_overall_cost(C, D, id1, a) - 10000*len(unassign_but_fea)
    return local_uti

#individual utility 2
def cal_local_uti_new(C,D,id1,fea_tasks,neighbors,a, all_players_action_copy):
    #players_all=copy.deepcopy(players_set)
    executing_tasks = a[:]
    for s in neighbors:
        if len(all_players_action_copy[s-1])>0:
            for neig_a in all_players_action_copy[s-1]:
                executing_tasks.append(neig_a)
    executing_tasks=set(executing_tasks) 
    remain_tasks=set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-executing_tasks
    unassign_but_fea=set(fea_tasks) & remain_tasks
    local_uti= -cal_overall_cost(C, D, id1, a) - 10000*len(unassign_but_fea)
    return local_uti


#Calculate the best action and regret value    
def cal_B_and_R(C,D,players_set):
    for u in players_set: 
        Best_a=[u['action_set'][0]]
        Best_uti=cal_local_uti(C, D, u['id'], u['feasible_tasks'], u['neighbor_set'], u['action_set'][0], players_set)
        for a in u['action_set'][1:]:
            uti=cal_local_uti(C, D, u['id'], u['feasible_tasks'], u['neighbor_set'], a, players_set)
            if uti > Best_uti:
                Best_a=[a]
                Best_uti=uti
            elif uti == Best_uti:
                Best_a.append(a)
        u['best_action']=random.choice(Best_a)
        current_uti=cal_local_uti(C, D, u['id'], u['feasible_tasks'], u['neighbor_set'], u['action'], players_set)
        regret=Best_uti-current_uti
        u['regret']=regret
    return players_set

def cal_nei_ave_regret(C,D,players_set):
    for s in players_set:
        sum_regret=0
        for z in s['neighbor_set']:
            sum_regret+=players_set[z-1]['regret']
        average_regret=sum_regret/len(s['neighbor_set'])
        s['deta_ave_regret']=s['regret']-average_regret 
    return players_set

def cal_better_action(C,D,players_set):
    for u in players_set:
        better_action=[]
        benchmark_uti=cal_local_uti(C, D, u['id'], u['feasible_tasks'], u['neighbor_set'], u['action'], players_set) #计算上一时刻对应效用
        for a in u['action_set']:
            uti=cal_local_uti(C, D, u['id'], u['feasible_tasks'], u['neighbor_set'], a, players_set)
            if uti >= benchmark_uti:
                better_action.append(a)
        u['better_action']=better_action
    return players_set

def all_equal(lst):
    return lst[1:] == lst[:-1]

def judge_satble(all_players_set):
    stable=True
    for d in all_players_set:
        if (not all_equal(d['memory_vector'])) or d['regret'] != 0:
            stable=False
    return stable
                
def record_action(players_set):
    action_record=[]
    for c in players_set:
        action_record.append(c['action'])    
    return action_record           
            
        
