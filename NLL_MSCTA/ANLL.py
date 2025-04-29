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

player_id = []
player_a_num = [] 
all_sat_plyers_set = []


for i in range(20):
    sat_players_set=intial_action_code(sat_players_set)  #initialize action encoding and return player set
    all_sat_plyers_set.append(copy.deepcopy(sat_players_set))
    t1=time.time()
    t=1
    all_glo_obj=[]
    
    a_player_id = [] #record the player's ID for ARLL comparison experiment
    a_player_a_num = [] #record the number of neighboring actions
    
    while True:
        global_obj=cal_global_obj_func(C, D, sat_players_set) 
        all_glo_obj.append(global_obj)
        
        sat_player = random.choice(sat_players_set) #randomly select a player, update asynchronously
        a_player_id.append(sat_player['id'])
        
        ########generate the neighborhood candidate action set########
        trial_code=[]
        while sat_player['action_code'] not in trial_code:
            trial_code=[]
            for i in range(1):  #ni=1
                p=random.randint(1, 1000)/1000 
                if p<=0.9:
                    cur_code=copy.deepcopy(sat_player['action_code'])
                    #######generate new encoding based on current action encoding#######
                    trial_last_code1=change_code_1(cur_code) 
                    trial_last_code2=change_code_2(cur_code)
                    trial_last_code=trial_last_code1 + trial_last_code2
                    trial_code+=trial_last_code
                else: 
                    current_code=copy.deepcopy(sat_player['action_code']) 
                    anoth_code=[random.choice([0,1]) for r in range(len(sat_player['feasible_tasks']))] #随便产生一个新动作的编码
                    while anoth_code == current_code:
                        anoth_code=[random.choice([0,1]) for r in range(len(sat_player['feasible_tasks']))]
                    trial_anoth_code1 = change_code_1(anoth_code) 
                    trial_anoth_code2 = change_code_2(anoth_code) 
                    trial_anoth_code = trial_anoth_code1 + trial_anoth_code2
                    trial_code += trial_anoth_code

        all_trial_code=[]
        [all_trial_code.append(i) for i in trial_code if i not in all_trial_code]       
        #######decoding#######
        all_candiate_act=[]
        for code1 in all_trial_code:
            all_candiate_act.append(decode(code1,sat_player['feasible_tasks']))
        a_player_a_num.append(len(all_candiate_act))   
        #######log-linear########
        uti_of_all_action=[]
        for m in all_candiate_act:
            uti = cal_local_uti(C, D, sat_player['id'], sat_player['feasible_tasks'], sat_player['neighbor_set'], m, sat_players_set)
            uti_of_all_action.append(uti)
        uti_of_all_action = np.array(uti_of_all_action) 
        prob_all_act = get_prob_LL(uti_of_all_action,90)
        next_t_action = get_a_action(prob_all_act, all_candiate_act)    

        new_code = [1 if sat_player['feasible_tasks'][i] in next_t_action else 0 for i in range(len(sat_player['feasible_tasks']))]
        sat_player['action_code'] = new_code

        sat_player['action']=next_t_action 
            
            
        if cal_global_obj_func(C, D, sat_players_set)==2594:
            all_glo_obj.append(cal_global_obj_func(C, D, sat_players_set))
            print(record_action(sat_players_set))
            break
            
        if  t>=50000:
            break
        t+=1
   
    plt.xlabel(r'Iteration $\it{k}$')
    plt.ylabel(r'Global objective $\it{F}$',rotation = 90)
    plt.plot(all_glo_obj)
    deta_t=time.time()-t1
    F_record.append(all_glo_obj)
    t_record.append(deta_t)
    action_profile.append(record_action(sat_players_set))
    
    all_glo_obj_min.append(min(all_glo_obj))
    
    player_id.append(a_player_id) #++++++
    player_a_num.append(a_player_a_num) #++++++
    
#np.save("C:/Users/am is are/Desktop/draft_data/ANLL_F50000",F_record) 
#plt.savefig('C:/Users/am is are/Desktop/draft_figure/ANLL50000.pdf',dpi=1200,bbox_inches='tight')
print(all_glo_obj_min)
print(np.mean(all_glo_obj_min))
print(t_record)
print(np.mean(t_record))
print('The standard deviation of the minimum F for all rounds is {}'.format(np.std(all_glo_obj_min,ddof=1)))

#np.save("C:/Users/am is are/Desktop/draft_data/ANLL_id_anum/ANLL_id",player_id) 
#np.save("C:/Users/am is are/Desktop/draft_data/ANLL_id_anum/ANLL_anum",player_a_num) 
#np.save("C:/Users/am is are/Desktop/draft_data/ANLL_id_anum/players_sats",all_sat_plyers_set)


    