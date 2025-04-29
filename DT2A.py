# -*- coding: utf-8 -*-
"""
DT2A

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
from draft_func import cal_B_and_R
from draft_func import judge_satble
from draft_func import record_action
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
for i in range(20):
    sat_players_set=intial_(sat_players_set, 100)  #initialize the player set
    t=1
    all_glo_obj=[]
    t1=time.time()
    while True:
        sat_players_set=cal_B_and_R(C, D, sat_players_set) #update each player's regret value and best action
        global_obj=cal_global_obj_func(C, D, sat_players_set)
        all_glo_obj.append(global_obj)
        for x in range(len(sat_players_set)):
            sat_player=sat_players_set[x]
            status_innovator=True
            if sat_player['regret']>0:
                for m in sat_player['neighbor_set']:
                    if sat_players_set[m-1]['regret'] > sat_player['regret']:
                        status_innovator=False
                    if sat_players_set[m-1]['regret']==sat_player['regret'] and sat_player['id']>sat_players_set[m-1]['id']:
                        status_innovator=False
            elif sat_player['regret']==0:
                status_innovator=False
            
            if status_innovator:
                new_action=sat_player['best_action']
            else:
                new_action=sat_player['action']
          
            del sat_players_set[x]['memory_vector'][0]
            sat_players_set[x]['memory_vector'].append(new_action)
            next_t_action=random.choice(sat_players_set[x]['memory_vector'])
            sat_players_set[x]['action']=next_t_action 
        if judge_satble(sat_players_set)==True:
            t2=time.time()-t1
            break
        t+=1
    F_record.append(all_glo_obj)
    t_record.append(t2)
    action_profile.append(record_action(sat_players_set))
   
    plt.xlabel(r'Iteration $\it{k}$')
    plt.ylabel(r'Global objective $\it{F}$',rotation = 90)
    plt.plot(all_glo_obj)
    
final_F=[]
min_F=[]
for i in F_record:
    final_F.append(i[-1])
    min_F.append(min(i))
print('The convergent F value is {}'.format(final_F))
print(np.mean(final_F))
print('The minimum F value is {}'.format(min_F))
print(np.mean(min_F))
print('The standard deviation of the minimum F for all rounds is {}'.format(np.std(min_F,ddof=1)))#样本标准差ddof=1.ddof=0为总体标准差 
print('Convergence time:',t_record)
print(np.mean(t_record)) 
#np.save("C:/Users/am is are/Desktop/draft_data/DT2A_F",F_record) 
#plt.savefig('C:/Users/am is are/Desktop/draft_figure/DT2A.pdf',dpi=1200,bbox_inches='tight')