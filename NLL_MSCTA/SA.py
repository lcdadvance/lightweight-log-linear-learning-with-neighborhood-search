# -*- coding: utf-8 -*-
"""

@author: am is are
"""

import numpy as np
import copy
import random
import time
import math
from draft_func import get_players
from draft_func import get_neighbors
from draft_func import intial_
from draft_func import cal_global_obj_func
from draft_func import cal_local_uti
from draft_func import cal_B_and_R
from draft_func import judge_satble
from draft_func import record_action
from satellite_task_alloction.func_LL import get_prob_LL
from satellite_task_alloction.func_LL import get_a_action
from satellite_task_alloction.func_LL import add_action
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False 

plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12}) 

inf=float('inf') 


C_test=np.array([[52,93,15,72,61,21,83,87,inf,75,inf,100,24,3,22],[inf,2,88,30,38,inf,inf,60,inf,33,76,58,22,89,inf],
            [inf,59,42,92,60,inf,inf,inf, 62, inf, inf, 51, 55, 64, inf],[inf, 7, inf, 73, 39, 18, 4, inf, inf, inf, inf, inf, 53, 2, 84]])
D_test=np.array([[-2.2, -32.2, 32.7, -25.7, -27.1, -29.2, -27, 30.8, inf, 30.8,inf,30.3, 27.1, -15.5, -33.7],[inf,7.1,29.1,6.5,-30.9,inf,inf,27.5,inf,7.6,31.6,32.9,10,-33.2,inf],
            [inf,-7.9,26.5,34.2,30.8,inf,inf,inf, -31.4, inf, inf, 31.1, -2.6, 32.3, inf],[inf, -25, inf, 30.3, 25.1, -31.3, -22.1, inf, inf, inf, inf, inf, 33.9, 26.1, 19.4]])

sat_players_set=get_players(C_test)
sat_players_set=get_neighbors(sat_players_set)
action_num=[] 
for x in sat_players_set:
    action_num.append(len(x['action_set']))

def get_fea_sat(C):
    info=[]
    for i in range(len(C[0])):
        sat_info={}
        sat_info['task_id']=i+1
        fea_sat=[]
        for j in range(len(C)):
            if C[j][i]<float('inf'):
                fea_sat.append(j+1) 
        sat_info['fea_sat']=fea_sat
        info.append(sat_info)
    return info
feasible_sat = get_fea_sat(C_test) 
print(feasible_sat)  
 
#######fitness function######## 
def func3(x):
    backup_players_set=copy.deepcopy(sat_players_set) 
    #print(sat_chosen)
    #index0 = [i+1 for i,val in enumerate(x) if val==0] 
    index1 = [i+1 for i,val in enumerate(x) if val==1]
    index2 = [i+1 for i,val in enumerate(x) if val==2]
    index3 = [i+1 for i,val in enumerate(x) if val==3]
    index4 = [i+1 for i,val in enumerate(x) if val==4]
    backup_players_set[0]['action'] =index1 
    backup_players_set[1]['action'] =index2
    backup_players_set[2]['action'] =index3
    backup_players_set[3]['action'] =index4
    action_chosen=[index1,index2,index3,index4]
    #print(action_chosen)
    fit = cal_global_obj_func(C_test, D_test, backup_players_set) #calculate individual fitness
    return fit , action_chosen

#####initialize###########
T=100  #initial temperature
L=200  #length of Markov chains
K=0.97 #temperature decay parameter
m=15   #number of tasks

t0=time.time()
#######Set initial value#######
#current_x=[random.choice([0,1,2,3,4,5]) for a in range(m)]
current_x=[]
for x in feasible_sat:
    current_x.append(random.choice(x['fea_sat']))
#######calculate fitness#######
f1=func3(current_x)[0]  
f=[f1]  
action_profile=[func3(current_x)[1]] 
while T > 0.001:
    for i in range(L):
        current_f=func3(current_x)[0]  
        #######generate random perturbations (generate new solutions based on old solutions)
        text_x=copy.deepcopy(current_x)
        #text_x[random.choice(range(m))]=random.choice([0,1,2,3,4,5])
        random_task_id = random.choice(range(m))
        text_x[random_task_id]=random.choice(feasible_sat[random_task_id]['fea_sat'])
        new_f=func3(text_x)[0]
        delta_f=new_f-current_f
        if delta_f <= 0:
            current_x = text_x
        else:
            if random.random() < np.exp(-delta_f/T):
                current_x = text_x
                
        f.append(func3(current_x)[0])
        action_profile.append(func3(current_x)[1])
    #######temperature decrease###########
    T=T*K
    print(T)

plt.xlabel(r'Iteration $\it{k}$')
plt.ylabel('Fitness',rotation = 90)
plt.plot(f)
print(min(f)) 
print(action_profile[f.index(min(f))])     
print(time.time()-t0)
#plt.savefig('C:/Users/am is are/Desktop/draft_figure/SA_35du.pdf',dpi=1200,bbox_inches='tight')