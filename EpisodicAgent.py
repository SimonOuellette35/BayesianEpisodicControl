from EpisodicMemory import EpisodicMemory

import numpy as np

class EpisodicAgent:

    def __init__(self, action_space):
        self.memory = EpisodicMemory(action_space)
        self.action_space = action_space

    def act(self, state):

        max_value = -9999
        max_a = None

        if self.memory.isEmpty():
            print("memory is empty, returning first action: ", self.action_space[0])
            return self.action_space[0]
        else:
            print("memory is not empty!")

        traces = {}
        for a in self.action_space:
            # first determine the current best expected action
            trace = self.memory.queryMemory(a, state)
            traces[a] = trace
            value = np.mean(trace)

            if value > max_value:
                max_value = value
                max_a = a

        return max_a

    def memorize(self, state, action, reward):
        self.memory.addMemory(state, action, reward)
