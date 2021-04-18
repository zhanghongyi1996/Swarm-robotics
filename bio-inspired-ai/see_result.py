import numpy as np
import math
import random
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy
from scipy.spatial.distance import cdist, pdist, euclidean
import seaborn as sns

a = np.loadtxt('trial1.txt')
b= np.loadtxt('trial2.txt')
print(a)
print(b)
c=(a+b)/2
print(c)

plt.figure(1)
sns.heatmap(c,annot=True, cmap='Reds')
plt.xlabel('bias',fontsize=20, color='k')
plt.ylabel('robot',fontsize=20, color='k')
plt.title('time consumption of small bias')


d= np.loadtxt('trial3_start_from 80 robots.txt')
plt.figure(2)
sns.heatmap(d,annot=True, cmap='Reds')
plt.xlabel('bias',fontsize=20, color='k')
plt.ylabel('robot',fontsize=20, color='k')
plt.title('time consumption of large range bias')

plt.show()