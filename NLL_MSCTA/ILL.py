# -*- coding: utf-8 -*-
"""

@author: am is are
"""

import numpy as np
import copy
import random
import time
from draft_func import get_players
from draft_func import get_neighbors
from draft_func import intial_
from draft_func import cal_global_obj_func
from draft_func import cal_local_uti_new
from draft_func import cal_B_and_R
from draft_func import judge_satble
from draft_func import record_action
from func_LL import get_prob_LL
from func_LL import get_a_action
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12})


inf=float('inf') #

C=np.array([[52,93,15,72,61,21,83,87,inf,75,inf,100,24,3,22],[inf,2,88,30,38,inf,inf,60,inf,33,76,58,22,89,inf],
            [inf,59,42,92,60,inf,inf,inf, 62, inf, inf, 51, 55, 64, inf],[inf, 7, inf, 73, 39, 18, 4, inf, inf, inf, inf, inf, 53, 2, 84]])
D=np.array([[-2.2, -32.2, 32.7, -25.7, -27.1, -29.2, -27, 30.8, inf, 30.8,inf,30.3, 27.1, -15.5, -33.7],[inf,7.1,29.1,6.5,-30.9,inf,inf,27.5,inf,7.6,31.6,32.9,10,-33.2,inf],
            [inf,-7.9,26.5,34.2,30.8,inf,inf,inf, -31.4, inf, inf, 31.1, -2.6, 32.3, inf],[inf, -25, inf, 30.3, 25.1, -31.3, -22.1, inf, inf, inf, inf, inf, 33.9, 26.1, 19.4]])
sat_players_set=get_players(C)
sat_players_set=get_neighbors(sat_players_set)


F_record=[]
t_record=[]
for i in range(20):
    t1=time.time()
    t=1
    all_glo_obj=[]
    all_t=[]
    sat_players_set=intial_(sat_players_set, 1) #initialize the player set
    #print(sat_players_set=cal_B_and_R(C, D, sat_players_set))
    while True:
        global_obj=cal_global_obj_func(C, D, sat_players_set) 
        all_glo_obj.append(global_obj)
        all_t.append(time.time()-t1)
        
        all_players_copy_action = [w['action'].copy() for w in sat_players_set]
        
        for x in range(len(sat_players_set)):
            a_player=sat_players_set[x] 
            w=random.randint(1, 1000)/1000
            if w<=0.1: #exploration probability
                indiv_uti_all_action=[]  
                for a in a_player['action_set']:
                    uti = cal_local_uti_new(C, D, a_player['id'], a_player['feasible_tasks'], a_player['neighbor_set'], a, all_players_copy_action)
                    indiv_uti_all_action.append(uti)
                indiv_uti_all_action = np.array(indiv_uti_all_action)
                prob_all_act = get_prob_LL(indiv_uti_all_action, 90)
                new_action = get_a_action(prob_all_act, a_player['action_set'])
                sat_players_set[x]['action']=new_action 
            else:
                pass
        if cal_global_obj_func(C, D, sat_players_set)==2594:
            all_glo_obj.append(cal_global_obj_func(C, D, sat_players_set))
            all_t.append(time.time()-t1)
            print(record_action(sat_players_set))
            break
        if t>=50000:
            break
        t+=1
    
    F_record.append(all_glo_obj)
    t_record.append(all_t)   
    plt.xlabel(r'Iteration $\it{k}$')
    plt.ylabel(r'Global objective $\it{F}$',rotation = 90)
    plt.plot(all_glo_obj)
   
#np.save("C:/Users/am is are/Desktop/draft_data/ISLL_F",F_record)  
#plt.savefig('C:/Users/am is are/Desktop/draft_figure/ISLL.pdf',dpi=1200,bbox_inches='tight')


##########data processing#############
F_min=[]
t_F_min=[]
t_last=[]
for a in F_record:
    F_min.append(min(a))
for j in range(len(F_min)):
    t_F_min.append(t_record[j][F_record[j].index(F_min[j])])
for b in t_record:
    t_last.append(b[-1])
    
print('The minimum F value is {}'.format(F_min))
print('The standard deviation of the minimum F for all rounds is {}'.format(np.std(F_min,ddof=1))) 
print('Average minimum F value: {}'.format(np.mean(F_min)))
print('The minimum F value corresponds to the time: {}'.format(t_F_min))
print('Average minimum F value corresponding time: {}'.format(np.mean(t_F_min)))
print('The time spent on each run is {}'.format(t_last))