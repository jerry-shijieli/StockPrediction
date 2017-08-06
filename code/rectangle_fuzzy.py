import pandas as pd
import numpy as np
import operator
from time import time

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def universe_partition(data, d1=10, d2=10):
    x_max, x_min = data.max(axis=0), data.min(axis=0)    
    std_val = data.std(axis=0)
    len_val = np.round(std_val / 10)
    u_max, u_min = int(x_max+d2), int(x_min-d1) # bound of universe discourse
    u_b = np.arange(u_min, u_max, step=float(len_val)) # cutting points
    u_discourse = u_b
    return u_discourse

def set_fuzzy_numbers(u_discourse_raw):
    u_s = u_discourse_raw[:-1]
    u_e = u_discourse_raw[1:]
    u_discourse = zip(u_s, u_e)
    fuzzy_numbers = list()
    for i, u_i in enumerate(u_discourse):
        fuzzy_numbers.append(np.array([u_i[0], u_i[1]]))
    return fuzzy_numbers

def membership_assignment(value_time_series, fuzzy_numbers):
    membership_list = np.digitize(value_time_series, fuzzy_numbers)-1
    membership_list[membership_list<0] = 0
    max_index = len(fuzzy_numbers) - 2
    membership_list[membership_list>max_index] = max_index
    return membership_list.tolist()

def get_membership(value, fuzzy_numbers):
    membership_index = np.digitize([value], fuzzy_numbers)[0]-1
    if membership_index < 0:
    	membership_index = 0
    max_index = len(fuzzy_numbers) - 2
    if membership_index > max_index:
    	membership_index = max_index
    return membership_index

def FLR(membership_time_series): # transition between consecutive observations
    transitions = list()
    for j, Aj in enumerate(membership_time_series):
        if j!=0:
            Ai = membership_time_series[j-1]
            transitions.append((Ai, Aj))
    return transitions

def FLR_weight(transitions, time_series): # compute jump frequency by FLR
    jumps = map(lambda x: x[1]-x[0], transitions) # compute jumps by transitions 
    jump_time_series = zip(jumps, time_series) # assign timestamp for each jump beta^t_p,p+k
    jump_counts = defaultdict(list) 
    for key, value in jump_time_series:
        jump_counts[key].append(value) # count jump by its timestamps
    jump_counts = {key: sum(value) for key, value in jump_counts.items()} # sum up total time for each jump
    total_count = float(sum(jump_counts.values()))
    for key, value in jump_counts.iteritems(): 
        jump_counts[key] = value / total_count # normalize jumps as weights
    return jump_counts

def FRG_weight(transitions, time_series): 
    transition_time_series = zip(transitions, time_series)
    transition_groups = map(lambda x: (x[0][0], (x[0][1], x[1])), transition_time_series) 
    transition_weights = defaultdict(list)
    for key, value in transition_groups:
        transition_weights[key].append(value) # group transitions by initial state A_i
    transition_weights = {key: dict(value) for key, value in transition_weights.items()}
    for key, value in transition_weights.iteritems():
        total_weight = float(sum(value.values()))
        value = {k: (v/total_weight) for k, v in value.items()} # normalize weight inside each group
        transition_weights[key] = value
    return transition_weights

# forecasting by fuzzy numbers
def fuzzy_add(A, B): # Proposition #1 (1)
    return A + B # A and B are numpy array type

def fuzzy_scale(c, A): # Proposition #1 (2)
    cA = c*A # A is numpy array type
    if c>=0:
        return cA
    else:
        return cA[::-1]

def forecast_jump(i, s, A_list):
    jumps = s.keys() # possible jumps
    m = len(A_list) # number of fuzzy numbers in model
    sA_list = list()
    sk_list = list()
    Aip_list = list()
    sA = np.array([0.0]*len(A_list[0]))
    for k in jumps:
        ip = k+i
        if (ip>=0 and ip<m): # check if index is within range
            sk_list.append(s[k])
            Aip_list.append(A_list[ip])
    sk_list = np.array(sk_list) / sum(sk_list) # normalize locally
    for i in range(sk_list.size):
        sA_list.append(fuzzy_scale(sk_list[i], Aip_list[i]))
    if len(sA_list)>0:
        for sa in sA_list:
            sA = fuzzy_add(sA, sa)
    return sA
    
def forecast_transition(i, w, A_list):
    wA = np.array([0.0]*len(A_list[0])) # default FLG relation
    if i in w.keys():
        for kj, v in w[i].iteritems():
            wA = fuzzy_add(wA, fuzzy_scale(v, A_list[kj]))
    return wA

def forecast_price(As, Aw, gamma=0.1):
    if gamma<0 or gamma>1:
        raise ValueError("gamma should be between 0.0 and 1.0 (inclusive on both ends)")     
    wAi = fuzzy_scale(1-gamma, Aw)
    if (np.sum(wAi) == 0): # no FLR observed in history
        sAi = As
    else:
        sAi = fuzzy_scale(gamma, As)
    Ai_pred = fuzzy_add(sAi, wAi)
    return np.mean(Ai_pred)

