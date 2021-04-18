import numpy as np
import math
import random
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy
from scipy.spatial.distance import cdist, pdist, euclidean
import seaborn as sns


def readdata(route):
    with open(route, "r") as f:
         content = f.read()

    lines = [line.split(" ") for line in content.split("\n")]
    lst = []
    for line in lines:
        lst += line

    result=[]
    for i in range(0, len(lst)):
        if lst[i] != '':
           ccc = lst[i].replace('.', '')
           ddd = ccc.replace('[', '')
           eee = ddd.replace(']', '')
           result.append(int(eee))
    return result

result1=readdata('data1.txt')
print(result1)
predict_expected = np.zeros((34,1))
max_point = np.zeros((34))
min_point = np.zeros((34))
time_finish = np.zeros((34,10))
for i in range(0,34):
    for j in range(0,10):
        time_finish[i][j]=result1[i * 10 + j]
    predict_expected[i] = np.mean(time_finish[i])
    max_point[i] = np.max(time_finish[i])
    min_point[i] = np.min(time_finish[i])
data_points = 34
x = np.linspace(0, data_points - 1, num=data_points)
plt.plot(predict_expected, linewidth=3., label='estimated value')
plt.fill_between(x,min_point,max_point,facecolor='LightSteelBlue')
plt.xlabel('Increasement of robot',fontsize=20, color='k')
plt.ylabel('Time consumption',fontsize=20, color='k')
plt.title('Time of completion of 95 percentage work in 50m*10m area')

plt.show()




