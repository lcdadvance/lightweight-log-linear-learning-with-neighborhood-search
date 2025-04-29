# -*- coding: utf-8 -*-
"""

@author: am is are
"""
import numpy as np
import random 

def get_prob_LL(a,tao):
    c=np.max(a/tao)
    exp_a=np.exp(a/tao-c)
    sum_exp_a=np.sum(exp_a)
    prob=exp_a/sum_exp_a
    return prob

def get_prob_LL1(a,tao):
    c=np.max(a)
    exp_a=np.exp((a-c)/tao)
    sum_exp_a=np.sum(exp_a)
    prob=exp_a/sum_exp_a
    return prob

def get_a_action(prob_a,action_set):
    x=random.randint(1, 9999999999)
    y=x/10000000000
    prob=[0]
    for i in range(len(prob_a)-1):
        prob.append(np.sum(prob_a[:i+1]))
    prob.append(1)
    cor_action=0
    for j in range(len(prob_a)):
        if y>prob[j] and y<=prob[j+1]:
            cor_action=action_set[j]
            break
    return cor_action

def add_action(list1,action_set):
    action_random=[]
    index1=action_set.index(list1[0])
    index2=action_set.index(list1[1])
    if index1>index2:
        tem=index2
        index2=index1
        index1=tem
    if len(action_set) >= 6:
        if index1==0:
            if index2 == len(action_set)-1: 
                action_random.append(action_set[1])
                action_random.append(action_set[index2-1])
            elif index2-index1 == 1:
                action_random.append(action_set[-1])
                action_random.append(action_set[index2+1])
            elif index2-index1 == 2:
                action_random.append(action_set[index1+1])
                action_random.append(action_set[index2+1])
            else:
                action_random.append(action_set[index1-1])
                action_random.append(action_set[index1+1])
                action_random.append(action_set[index2-1])
                action_random.append(action_set[index2+1])
        else:
            if index2==len(action_set)-1:
                if index2-index1==1:    
                    action_random.append(action_set[0])
                    action_random.append(action_set[index1-1])
                elif index2-index1==2:
                    action_random.append(action_set[index1-1])
                    action_random.append(action_set[index1+1])
                elif index2-index1 >2:
                    action_random.append(action_set[index1-1])
                    action_random.append(action_set[index1+1])
                    action_random.append(action_set[index2-1])
            else:
                if index2-index1==1:
                    action_random.append(action_set[index1-1])
                    action_random.append(action_set[index2+1])
                elif index2-index2==2:
                    action_random.append(action_set[index1-1])
                    action_random.append(action_set[index1+1])
                    action_random.append(action_set[index2+1])
                else:
                    action_random.append(action_set[index1-1])
                    action_random.append(action_set[index1+1])
                    action_random.append(action_set[index2-1])
                    action_random.append(action_set[index2+1])
    action_random+=list1
    return action_random 



if __name__=='__main__':
    list1=np.array([1000,2000,20000,30000])
    prob1=get_prob_LL(list1, 1)
    print(prob1)
    print(np.exp(1)/(np.exp(1)+np.exp(2)+np.exp(3)+np.exp(4)))
    print(np.exp(2)/(np.exp(1)+np.exp(2)+np.exp(3)+np.exp(4)))
    print(np.exp(3)/(np.exp(1)+np.exp(2)+np.exp(3)+np.exp(4)))
    print(np.exp(4)/(np.exp(1)+np.exp(2)+np.exp(3)+np.exp(4)))
    sum_prob = get_a_action(prob1, ['a','b','c','d'])
    print(sum_prob)
