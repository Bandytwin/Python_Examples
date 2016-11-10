# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:10:11 2016

@author: Andrew

*** To improve performance add in an increase to the likelihood of choosing the
arm with a higher win-rate as time increases.

"""
from random import random
from bisect import bisect
import numpy as np

class SlotMachine:
    def __init__(self, win_prob):
        self.p = win_prob
        
    def play(self):
        return int(random() < self.p)
        
class Casino:
    def __init__(self, num_tokens, p_vec):
        self.tokens_left = num_tokens
        self.tokens_start = num_tokens
        self.arms = []
        for ii in range(len(p_vec)):
            self.arms.append(SlotMachine(p_vec[ii]))
            
    def reset(self):
        self.tokens_left = self.tokens_start
            
    def playMachine(self, id):
        if self.tokens_left == 0:
            self.reset()            
            return 'Out'
        else:
            self.tokens_left -= 1
            return self.arms[id].play()
            
class BayesianInference:
    def __init__(self, casino, alpha, beta):
        self.casino = casino
        self.alpha = alpha # Win reward multiplier
        self.beta = beta # Play penalty multiplier
        
        self.total_wins = 0
        self.arms_wins = np.zeros([len(casino.arms)])
        self.arms_plays = np.zeros([len(casino.arms)])
        self.arm_id_vec = []
    
    def chooseArm(self):
        u = (1 + self.alpha*self.arms_wins) / (1 + self.beta*self.arms_plays)
        u = u/sum(u)        
        u_cum = u.cumsum()
        bisector = random()
        arm_id = bisect(u_cum,bisector)
        
        return arm_id
    
    def play(self):
        while(1):
            arm_id = self.chooseArm()
            result = self.casino.playMachine(arm_id)
            if result == 'Out':
                break
            self.total_wins += result
            self.arms_plays[arm_id] += 1
            self.arms_wins[arm_id] += result
            self.arm_id_vec.append(arm_id)
        
tokens = 500
p_vec = [0.3, 0.5, 0.7]
c = Casino(tokens, p_vec)
player = BayesianInference(c, 4, 2)
player.play()
print("Arms played: ", player.arms_plays)
print("Total Winnings: ", player.total_wins)
plt.plot(player.arm_id_vec)
                
        
        