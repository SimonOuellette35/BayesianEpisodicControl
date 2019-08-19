from EpisodicMemoryV2 import EpisodicMemory
import numpy as np

mem = EpisodicMemory([0, 1, 2])

# Here the return is high and the risk is low, but we only have 3 samples so our estimation certainty is very low.
# This should not be the highest conviction action.
MU0 = 0.20
SD0 = 0.05

# This should be the highest conviction action, because we have plenty of samples (100), and the variance
# is relatively low for a relatively high return.
MU1 = 0.15
SD1 = 0.12

# Here we have plenty of samples to estimate the distribution, but it's a risky action even though on average it is
# better. Should be lower conviction than a1.
MU2 = 0.20
SD2 = 0.5

STATE_DIM = 3

for _ in range(3):
    s = np.random.normal(0, 0.2, STATE_DIM)
    r = np.random.normal(MU0, SD0)

    mem.addMemory(s, 0, r)

for _ in range(100):

    s = np.random.normal(0, 0.2, STATE_DIM)
    r = np.random.normal(MU1, SD1)

    mem.addMemory(s, 1, r)

for _ in range(100):

    s = np.random.normal(0, 0.2, STATE_DIM)
    r = np.random.normal(MU2, SD2)

    mem.addMemory(s, 2, r)

print("Action 0: ")
Q0_E_mu, Q0_std_mu, Q0_E_sd, conviction0 = mem.queryMemory(0, [0.04, -0.08, 0.015])

print("E[mu] = %s, Stdev[mu] = %s, E[sd] = %s" % (
    Q0_E_mu,
    Q0_std_mu,
    Q0_E_sd
))

print("Calculated conviction = ", conviction0)

print("Action 1: ")
Q1_E_mu, Q1_std_mu, Q1_E_sd, conviction1 = mem.queryMemory(1, [0.04, -0.08, 0.015])

print("E[mu] = %s, Stdev[mu] = %s, E[sd] = %s" % (
    Q1_E_mu,
    Q1_std_mu,
    Q1_E_sd
))

print("Calculated conviction = ", conviction1)

print("Action 2: ")
Q2_E_mu, Q2_std_mu, Q2_E_sd, conviction2 = mem.queryMemory(2, [0.04, -0.08, 0.015])

print("E[mu] = %s, Stdev[mu] = %s, E[sd] = %s" % (
    Q2_E_mu,
    Q2_std_mu,
    Q2_E_sd
))

print("Calculated conviction = ", conviction2)
