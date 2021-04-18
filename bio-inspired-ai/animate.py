import clean_window
import numpy as np
import math
import random
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy
from scipy.spatial.distance import cdist, pdist, euclidean
import warehouse
import pickle
import sys
import os

number_agents = 10
anim=0
limit=2000
trial = 10
rob_change = 20
time_finish = np.zeros((rob_change,trial))

for i in range(0,rob_change):
    for j in range(0,trial):
          time_finish[i][j]=clean_window.data(number_agents+i, anim, limit).counter
          print(time_finish)

np.savetxt("result.txt", time_finish);
