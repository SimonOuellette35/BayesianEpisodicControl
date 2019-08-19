import math
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

class EpisodicMemory:

    def __init__(self, action_space):
        self.is_init = False
        self.state_history = {}
        self.reward_history = {}
        self.action_space = action_space
        for a in action_space:
            self.state_history[a] = []
            self.reward_history[a] = []

    def isEmpty(self):
        if self.is_init:
            return False
        else:
            return True

    def addMemory(self, state, action, reward):
        self.state_history[action].append(state)
        self.reward_history[action].append(reward)

        # TODO: remove oldest entries

    def calculateDistance(self, s1, s2):
        dist = 0.
        for i in range(len(s1)):
            dist += (s1[i] - s2[i]) ** 2.0

        return math.sqrt(dist)

    # TODO: optimize this by operating on numpy vectors all at once, rather than 1 state at a time...
    def queryMemory(self, a, state, k=50):

        # 1) go through state_history, calculate distance scores with current state.
        dist_idx = []
        for s_idx in range(len(self.state_history[a])):
            s = self.state_history[a][s_idx]
            dist = self.calculateDistance(s, state)
            dist_idx.append([dist, s_idx])

        # 2) keep the k most similar
        dist_idx = np.array(sorted(dist_idx, key=lambda tup: tup[0]))
        indices = dist_idx[:k, 1]
        rewards = [self.reward_history[a][int(i)] for i in indices]
        #print("rewards: ", rewards)

        # 3) Bayesian model of a normal distribution's mu and sd params on this dataset
        # TODO: there has to be a computationally faster method of doing Bayesian estimation of a Gaussian...
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, sd=1.)
            sd = pm.Uniform('sd', 0.0001, 1.)

            pm.Normal("likelihood", mu=mu, sd=sd, observed=rewards)

            trace = pm.sample(500, tune=500, target_accept=0.9)

            pm.traceplot(trace)
            plt.show()

        # 4) Calculate conviction
        # TODO: include similarity weighting in this metric?
        estimate_mu = np.mean(trace['mu'])
        stdev_sd = np.std(trace['mu'])

        estimate_sd = np.mean(trace['sd'])

        conviction = estimate_mu / (stdev_sd * estimate_sd)

        return estimate_mu, stdev_sd, estimate_sd, conviction