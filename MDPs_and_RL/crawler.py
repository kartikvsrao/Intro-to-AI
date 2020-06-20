"""
Project 3 - MDPs and RL

Team Members :  Bennett Meares
                Kartik Rao
"""

###################################
###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 10/24/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec14.pdf?attachauth=ANoY7co0wnqGrNKk_dFBy81-YEceaP4nrpivfhqWLxtaCq5mZMteUmT5RTQFJx5qJjqtj32ww1ZuATozzjRnVcTkSAISDVrbuvP72CaEDp8121897ws6Wnq1oOrhVTOBlgJwkKI8TgVE5WYPWVE2NgogOJrs-KQdMHvuc5-Wjl_EU9jlppaMesC9nPa8c1SRXX7YiLTMUv20AZiwuu866-QNdxxZg42W7w%3D%3D&attredirects=0
"""
###################################
# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
#
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this file, you should test your Q-learning implementation on the robot crawler environment
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions


Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""

# use random library if needed
import random
import copy


###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 10/24/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec14.pdf?attachauth=ANoY7co0wnqGrNKk_dFBy81-YEceaP4nrpivfhqWLxtaCq5mZMteUmT5RTQFJx5qJjqtj32ww1ZuATozzjRnVcTkSAISDVrbuvP72CaEDp8121897ws6Wnq1oOrhVTOBlgJwkKI8TgVE5WYPWVE2NgogOJrs-KQdMHvuc5-Wjl_EU9jlppaMesC9nPa8c1SRXX7YiLTMUv20AZiwuu866-QNdxxZg42W7w%3D%3D&attredirects=0
"""
###################################


def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 3500000
    eps_diff = 0.00005
    eps_decay = 0.00000005
    #########################
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    for i in range(len(pi)):
        pi[i] = random.randint(0,len(env.trans_model[i]) - 1)
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    #Exploration vs Exploitation
    def choose_action(eps,policy,s):
        #Exploration
        if random.uniform(0.0,1.0) < eps:
            return random.randint(0,NUM_ACTIONS - 1)
        #Exploitation
        return policy[s]

    def take_action(s,a):
        t = env.trans_model[s][a]
        #action thresholds
        a_thresholds = [(0.0,0.0)] * len(t)
        i = 0
        i = 0
        #end threshold
        e_thresh = 0.0
        #beg threshold
        b_thresh = 0.0
        #for all actions
        for p,s_,r,term in t:
            e_thresh += p
            a_thresholds[i] = (b_thresh, e_thresh)
            b_thresh += p
            i += 1
        roll = random.uniform(0.0,0.9999999999999)
        i = 0
        #Pick state to land
        for thresh in a_thresholds:
            if roll >= thresh[0] and roll < thresh[1]:
                ## return s_ and r
                return t[0][1],t[0][2],t[0][3]

    def extract_pi(q):
        policy = [0] * NUM_STATES
        s = 0
        for s_list in q:
            policy[s] = s_list.index(max(s_list))
            s += 1
        return policy

    ## initial state s
    s = random.randint(0,NUM_STATES - 1)
    q = [[0.0] * 4 for i in range(NUM_STATES)]
    k = 1

    while k <= max_iterations and eps >= 0:
        alpha = float(1 / k)
        #Epsilon greedy
        a = choose_action(eps,pi,s)
        #Take action a from state s to reach s_
        s_,r,term = take_action(s,a)
        #If terminal, reward is the sample, get new random starting state
        if term:
            sample = r
            s_ = random.randint(0,NUM_STATES - 1)
            env.reset()
        else:
            #Calculate sample
            sample = r + (gamma * max(q[s_]))
        #Update q
        q[s][a] = ((1 - alpha) * q[s][a]) + (alpha * sample)
        #Update V
        v[s] = max(q[s])
        #Update Pi
        pi[s] = q[s].index(max(q[s]))
        #Update s
        s = s_
        #Decaying Epsilon
        eps -= eps_diff
        eps_diff *= (1 + eps_decay)
        k += 1

    pi = extract_pi(q)
    logger.log(k, v, pi)

###############################################################################
    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk

    algs = {
        "Q Learning": q_learning,
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()
