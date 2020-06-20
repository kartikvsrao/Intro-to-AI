# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) 
#
import sys
import random

import numpy as np

class GridWorld(object):

    def __init__(self, reward_map, starts=[0], goals=None, obstacles=[], noise=0):
        class Discrete(object):
            def __init__(self, n):
                self.n = n
            def sample(self):
                return np.random.randint(self.n)

        self.n_rows = len(reward_map)
        self.n_cols = len(reward_map[0])
        self.observation_space = Discrete(self.n_rows*self.n_cols)
        self.action_space = Discrete(4)
        self.starts = starts
        self.goals = [self.observation_space.n-1] if goals is None else goals
        self.obstacles = obstacles
        self.action_desc = ["Up", "Right", "Down", "Left"]
        self.reward_map = reward_map
        # build transition model
        def a2s(r, c, a):
            r_ = max(0, min(self.n_rows-1, r + ((a+1)%2)*(-1 if a == 0 else 1)))
            c_ = max(0, min(self.n_cols-1, c + (a%2)*(-1 if a == 3 else 1)))
            s_ = r_ * self.n_cols + c_
            return r_, c_, s_
        self.trans_model = [
            [None for a in range(self.action_space.n)]
            for s in range(self.n_rows*self.n_cols)
        ]
        for s in range(self.observation_space.n):
            r = int(s / self.n_cols)
            c = s % self.n_cols
            for a in range(self.action_space.n):
                if s in self.goals or s in self.obstacles:
                    self.trans_model[s][a] = [
                        (1.0, s, reward_map[r][c], True)
                    ]
                elif noise > 0:
                    self.trans_model[s][a] = []
                    for a_ in [a, (a+1)%self.action_space.n, (a+3)%self.action_space.n]:
                        p = (1-noise) if a_ == a else noise/2.0
                        r_, c_, s_ = a2s(r, c, a_)
                        if s_ in self.obstacles:
                            s_ = s
                        self.trans_model[s][a].append(
                            (p, s_, reward_map[r][c], False)
                        )
                else:
                    r_, c_, s_ = a2s(r, c, a)
                    if s_ in self.obstacles:
                        s_ = s
                    self.trans_model[s][a] = [
                        (1.0, s_, reward_map[r][c], False)
                    ]

        self.state = None  # init state
        self.last_action = None

    def step(self, a):
        w = [p for p, *_ in self.trans_model[self.state][a]]
        i = random.choices(range(len(w)), weights=w, k=1)[0]
        p, self.state, r, terminal = self.trans_model[self.state][a][i]
        self.last_action = a
        return self.state, r, terminal, {"prob": p}

    def reset(self):
        self.state = random.choices(self.starts)[0]
        self.last_action = None
        return self.state

    def render(self, outfile=sys.stdout):
        if self.last_action is None:
            string = ""
        else:
            string = "Action: " + self.action_desc[self.last_action] + "\n"
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = r*self.n_cols + c
                if s == self.state:
                    ch = "\x1b[31mA\x1b[0m"
                elif s in self.goals:
                    if self.reward_map[r][c] > 0:
                        ch = "G"
                    else:
                        ch = "x"
                else:
                    ch = "."
                if s in self.goals:
                    if self.reward_map[r][c] < 0:
                        ch = "\x1b[44m" + ch + "\x1b[0m"
                    else:
                        ch = "\x1b[41m" + ch + "\x1b[0m"
                string += ch
            string += "\n"
        outfile.write(string)
    
    def dump(self):
        if self.last_action is None:
            string = ""
        else:
            string = "Action: " + self.action_desc[self.last_action] + "\n"
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = r*self.n_cols + c
                if s == self.state:
                    ch = "A"
                elif s in self.goals:
                    if self.reward_map[r][c] > 0:
                        ch = "G"
                    else:
                        ch = "x"
                else:
                    ch = "."
                string += ch
            string += "\n"
        return string
    
    def __str__(self):
        return self.dump()
