"""
Project 3 - MDPs and RL

Team Members :  Bennett Meares
                Kartik Rao
"""

###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 10/22/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec11.pdf?attachauth=ANoY7cqBBVwVqHxKjnbOdEzgl8B3sseSxJz7atRw6xcBIjQ6TutzOYCfo8a8_B4yip41iWp-LKY9hmkTKd7hsMyB-7P9VCHuYDjkTVWfeNGLLQhoizbmWO11XCFDWilFHCuF4LUf0KNfb-8d87MABJsJTFK_byI6_jSI2XMzgPDXSuBFfhqoQ7GlRuRzeB6R-Hi0N1z8LNuz4S01bb1Qya6N3hyvxbj0jQ%3D%3D&attredirects=0
"""
###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 10/24/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec14.pdf?attachauth=ANoY7co0wnqGrNKk_dFBy81-YEceaP4nrpivfhqWLxtaCq5mZMteUmT5RTQFJx5qJjqtj32ww1ZuATozzjRnVcTkSAISDVrbuvP72CaEDp8121897ws6Wnq1oOrhVTOBlgJwkKI8TgVE5WYPWVE2NgogOJrs-KQdMHvuc5-Wjl_EU9jlppaMesC9nPa8c1SRXX7YiLTMUv20AZiwuu866-QNdxxZg42W7w%3D%3D&attredirects=0
"""
###################################
# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
#
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement three classic algorithms for
solving Markov Decision Processes either offline or online.
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments.
You will also have the opportunity to use Q-learning to control a simulated robot
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import copy

###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 10/22/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec11.pdf?attachauth=ANoY7cqBBVwVqHxKjnbOdEzgl8B3sseSxJz7atRw6xcBIjQ6TutzOYCfo8a8_B4yip41iWp-LKY9hmkTKd7hsMyB-7P9VCHuYDjkTVWfeNGLLQhoizbmWO11XCFDWilFHCuF4LUf0KNfb-8d87MABJsJTFK_byI6_jSI2XMzgPDXSuBFfhqoQ7GlRuRzeB6R-Hi0N1z8LNuz4S01bb1Qya6N3hyvxbj0jQ%3D%3D&attredirects=0
"""
###################################
def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.
        In this case, you may want to exit the algorithm earlier. A way to check
        if value iteration has already converged is to check whether
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging

### Please finish the code below ##############################################
###############################################################################
    #initalize old states to negatice number so it doesnt affect convergence.
    v_old = [1e-4] * NUM_STATES
    terms = []
    terms_rewards = {}
    q_values = {}
    ## up to H iterations
    k = 1
    d = 1
    #setting min_iterations as 4 if max iterations is greater than 4. is its lesser or equal to 4 then the max and min iterations are the same.
    if max_iterations > 4:
        min_iterations = 4
    else:
        min_iterations = max_iterations
    #Condition for convergencer
    while k <= max_iterations and abs(sum(v) - sum(v_old)) >= 1e-4 or k < min_iterations:
        ## for all states
        v_old = copy.deepcopy(v)
        for s in range(NUM_STATES):
            ## find maximum value from all actions
            v_actions = [0,0,0,0]
            ### for all actions (up, right, down, left)
            for a in range(len(env.trans_model[s])):
                ## t contains three outcomes:
                ##   - 80% for correct outcome
                ##   - 10% each for two incorrect outcomes (i.e. noise)
                t = env.trans_model[s][a]
                val = 0
                for p,s_,r,term in t:
                    if term and k == 1:
                        terms.append(s_)
                        terms_rewards[s_] = r
                    val += p * (r + (gamma * v_old[s_]))
                    #  print('k:',k,'\ns:',s,'\na:',a)
                    #  print('p:',p,'\ns_:',s_,'\nr:',r,'\nterm:',term)
                    #  print('val:',val)
                    #  input()
                v_actions[a] = val
            ## choose value for s from best action
            q_values[s] = v_actions
            v[s] = max(q_values[s])
            ## assign best action to policy
            pi[s] = v_actions.index(v[s])
            if s in terms:
                v[s] = terms_rewards[s]

        logger.log(k, v, pi)
        k += 1


###############################################################################
    return pi

###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 10/22/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec11.pdf?attachauth=ANoY7cqBBVwVqHxKjnbOdEzgl8B3sseSxJz7atRw6xcBIjQ6TutzOYCfo8a8_B4yip41iWp-LKY9hmkTKd7hsMyB-7P9VCHuYDjkTVWfeNGLLQhoizbmWO11XCFDWilFHCuF4LUf0KNfb-8d87MABJsJTFK_byI6_jSI2XMzgPDXSuBFfhqoQ7GlRuRzeB6R-Hi0N1z8LNuz4S01bb1Qya6N3hyvxbj0jQ%3D%3D&attredirects=0
"""
###################################
def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations.
        In this case, you should exit the algorithm. A simple way to check
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges
        very fast and policy evaluation should end upon convergence. A way to check
        if policy evaluation has converged is to check whether the max over (or sum of)
        L1 or L2 norm between the values before and after an iteration is small enough.
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy;
        here you can update the visualization of value by simply calling logger.log(i, v).

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    def evaluate_policy(policy):
        k = 1

        v = [0.0] * NUM_STATES
        v_old = copy.deepcopy(v)
        if max_iterations > 4:
            min_iterations = 4
        else:
            min_iterations = max_iterations
        d = 1

        while k <= max_iterations and d >= 1e-4 or k < min_iterations:
            d = abs(sum(v) - sum(v_old))
            v_old = copy.deepcopy(v)
            ## for all states
            for s in range(NUM_STATES):
                ## find maximum value from all actions
                v_actions = [0,0,0,0]
                ## take action a (prescribed by policy)
                a = policy[s]
                t = env.trans_model[s][a]
                val = 0
                ## for every outcome of a
                for p,s_,r,term in t:
                    ## capture inital values for terminal states
                    if term and k == 1:
                        terms.append(s_)
                        terms_rewards[s_] = r
                    ## apply Bellman equation
                    val += p * (r + (gamma * v_old[s_]))
                v[s] = val
                ## reset value of terminal states to inital reward
                if s in terms:
                    v[s] = terms_rewards[s]
            k += 1
        return v

    old_pi = [-1] * NUM_STATES
    terms = []
    terms_rewards = {}
    k = 1

    while k <= max_iterations and old_pi != pi:
        v = evaluate_policy(pi)
        old_pi = copy.deepcopy(pi)

        ## improve pi
        for s in range(NUM_STATES):

           ## find maximum value from all actions
           v_actions = [-1,-1,-1,-1]
           for a in range(len(env.trans_model[s])):
                t = env.trans_model[s][a]
                val = 0
                for p,s_,r,term in t:
                    val += p * (r + (gamma * v[s_]))
                v_actions[a] = val

           # update policy to best action for state s
           pi[s] = v_actions.index(max(v_actions))
        logger.log(k, v, pi)
        k += 1

###############################################################################
    return pi

###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 10/24/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec14.pdf?attachauth=ANoY7co0wnqGrNKk_dFBy81-YEceaP4nrpivfhqWLxtaCq5mZMteUmT5RTQFJx5qJjqtj32ww1ZuATozzjRnVcTkSAISDVrbuvP72CaEDp8121897ws6Wnq1oOrhVTOBlgJwkKI8TgVE5WYPWVE2NgogOJrs-KQdMHvuc5-Wjl_EU9jlppaMesC9nPa8c1SRXX7YiLTMUv20AZiwuu866-QNdxxZg42W7w%3D%3D&attredirects=0
"""
###################################
def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################
    eps_decay = 0.00000005
    eps_diff = 0.00001

### Please finish the code below ##############################################
###############################################################################
    #Exploration vs Exploitation
    def choose_action(eps,policy,s):
        #Exploration
        if random.uniform(0.0,1.0) < eps:
            return random.randint(0,len(env.trans_model[s]) - 1)
        #Exploitation
        return policy[s]

    #Choose action
    def take_action(s,a):
        t = env.trans_model[s][a]
        #action thresholds
        a_thresholds = [(0.0,0.0)] * len(t)
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
                return t[i]

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
    ## initalize random policy
    for i in range(len(pi)):
        pi[i] = random.randint(0,len(env.trans_model[i]) - 1)
    while k <= max_iterations:
        alpha = float(1 / k)
        #Epsilon greedy
        a = choose_action(eps,pi,s)
        #Take action a from state s to reach s_
        p,s_,r,term = take_action(s,a)
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
        #Uncomment the below line to visualize the state values
        #  logger.log(k, v, pi)
        #Decaying Epsilon
        eps -= eps_diff
        eps_diff *= (1 + eps_decay)
        k += 1

    #Extract pi
    pi = extract_pi(q)
    logger.log(k, v, pi)
###############################################################################
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()
