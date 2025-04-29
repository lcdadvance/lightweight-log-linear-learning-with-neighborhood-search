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
from draft_func import cal_global_obj_func
from draft_func import cal_local_uti
from draft_func import record_action
from func_LL import get_prob_LL
from func_LL import get_a_action
from draft_func1 import intial_action_code
from draft_func1 import change_code_1
from draft_func1 import change_code_2
from draft_func1 import decode
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12})

inf=float('inf') 


C=np.array([[52,93,15,72,61,21,83,87,inf,75,inf,100,24,3,22],[inf,2,88,30,38,inf,inf,60,inf,33,76,58,22,89,inf],
            [inf,59,42,92,60,inf,inf,inf, 62, inf, inf, 51, 55, 64, inf],[inf, 7, inf, 73, 39, 18, 4, inf, inf, inf, inf, inf, 53, 2, 84]])
D=np.array([[-2.2, -32.2, 32.7, -25.7, -27.1, -29.2, -27, 30.8, inf, 30.8,inf,30.3, 27.1, -15.5, -33.7],[inf,7.1,29.1,6.5,-30.9,inf,inf,27.5,inf,7.6,31.6,32.9,10,-33.2,inf],
            [inf,-7.9,26.5,34.2,30.8,inf,inf,inf, -31.4, inf, inf, 31.1, -2.6, 32.3, inf],[inf, -25, inf, 30.3, 25.1, -31.3, -22.1, inf, inf, inf, inf, inf, 33.9, 26.1, 19.4]])

sat_players_set=get_players(C)
sat_players_set=get_neighbors(sat_players_set)


F_record=[] 
t_record=[] 
action_profile=[]
all_glo_obj_min=[]

player_id=np.load("C:/Users/am is are/Desktop/draft_data/ANLL_id_anum/ANLL_id.npy",allow_pickle=True)
player_a_num=np.load("C:/Users/am is are/Desktop/draft_data/ANLL_id_anum/ANLL_anum.npy",allow_pickle=True)
player_sats=np.load("C:/Users/am is are/Desktop/draft_data/ANLL_id_anum/players_sats.npy",allow_pickle=True)

for i in range(20):
    sat_players_set=copy.deepcopy(player_sats[i])
    t1=time.time()
    t=0 
    all_glo_obj=[]
    
    a_player_id = player_id[i]
    print(len(a_player_id))
    a_player_a_num = player_a_num[i] 

    for x in range(len(a_player_id)):
        global_obj=cal_global_obj_func(C, D, sat_players_set) 
        all_glo_obj.append(global_obj)
        
        sat_player = sat_players_set[a_player_id[t]-1] 
        a_num = a_player_a_num[t] 
        
        trial_action = random.sample(sat_player['action_set'],a_num)
         
        #######log-linear########
        uti_of_all_action=[]
        for m in trial_action:
            uti = cal_local_uti(C, D, sat_player['id'], sat_player['feasible_tasks'], sat_player['neighbor_set'], m, sat_players_set)
            uti_of_all_action.append(uti)
        uti_of_all_action = np.array(uti_of_all_action) 
        prob_all_act = get_prob_LL(uti_of_all_action,90)
        next_t_action = get_a_action(prob_all_act, trial_action)    
        sat_player['action']=next_t_action
            
        t+=1
   
    plt.xlabel(r'Iteration $\it{k}$')
    plt.ylabel(r'Global objective $\it{F}$',rotation = 90)
    plt.plot(all_glo_obj)
    deta_t=time.time()-t1
    F_record.append(all_glo_obj)
    t_record.append(deta_t)
    action_profile.append(record_action(sat_players_set))
    
    all_glo_obj_min.append(min(all_glo_obj))
    
#np.save("C:/Users/am is are/Desktop/draft_data/ANLL_random_F50000",F_record)
#plt.savefig('C:/Users/am is are/Desktop/draft_figure/ANLL_random50000.pdf',dpi=1200,bbox_inches='tight')
print(all_glo_obj_min)
print(np.mean(all_glo_obj_min))
print(t_record)
print(np.mean(t_record))
print('The standard deviation of the minimum F for all rounds is {}'.format(np.std(all_glo_obj_min,ddof=1)))


    
