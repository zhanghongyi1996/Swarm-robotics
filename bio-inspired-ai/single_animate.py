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


number_agents = 80
anim=0
limit=2000
trial = 10
vertical_speed = 0.5
rob_change_time = 9
speed_change_time= 8
time_finish = np.zeros((rob_change_time,speed_change_time))

for i in range(0,rob_change_time):
    for j in range(0,speed_change_time):
          time_finish[i][j]=clean_window.data(number_agents + 10 * i, anim, limit, vertical_speed + 0.2 * j).counter
          print(time_finish)


np.savetxt("trial3_start_from 80 robots.txt", time_finish)
