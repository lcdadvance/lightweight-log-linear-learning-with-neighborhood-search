# -*- coding: utf-8 -*-
"""

@author: am is are
"""

import numpy
import random
import copy

#Initialize action encoding
def intial_action_code(players_set):
    for x in players_set:
        L=len(x['feasible_tasks'])
        intial_action_code=[random.choice([0,1]) for r in range(L)]
        x['action_code']=intial_action_code
        #######decoding#########
        x['action']=[x['feasible_tasks'][i] for i in range(L) if intial_action_code[i]==1]
    return players_set 


def intial_Tabu(players_set):
    for x in players_set:
        L=len(x['feasible_tasks'])
        x['Tabu_list']=[0 for i in range(L)]
    return players_set

# swap operator
def change_code_1(act_code):
    trial_code=[act_code]  
    if sum(act_code)==0:
        return trial_code
    else:
        index_of_0 = [i for i,val1 in enumerate(act_code) if val1==0] 
        index_of_1 = [j for j,val2 in enumerate(act_code) if val2==1]
    
        for x in index_of_1:
            for y in index_of_0:
                back_code=copy.deepcopy(act_code)
                back_code[x] = 0  
                back_code[y] = 1  
                trial_code.append(back_code)
                
        return trial_code

# increment-decrement operator    
def change_code_2(act_code):
    trial_code=[]
    for b in range(len(act_code)):
        back_code1=copy.deepcopy(act_code)
        back_code1[b]=0 if back_code1[b]==1 else 1
        trial_code.append(back_code1)
    return trial_code

def decode(the_code,feasible_tasks):
    L=len(feasible_tasks)
    return [feasible_tasks[i] for i in range(L) if the_code[i]==1]

if __name__=='__main__':
    all_code=[1,0,0,1,1]
    feasible_task=[1,3,5,7,8]
    act=decode(all_code,feasible_task)
    print(act)
    