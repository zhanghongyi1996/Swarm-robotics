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
import for_animation_show


number_agents=30
anim=1
limit=2000

for_animation_show.data(number_agents,anim,limit)