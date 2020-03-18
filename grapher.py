# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
import sys

results = sys.argv[1]

filename = sys.argv[2]

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{0:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

df = pd.read_csv(results, sep=" ", header=None, names=np.arange(10))

classes = df.iloc[:9,4].values
percs = np.array(df.iloc[:9,8].values,dtype='float32')

plt.figure(figsize=(18, 10))

index = np.arange(len(classes))
rects = plt.bar(index, percs, 0.4, label='mAP')
plt.xlabel('category', fontsize=14)
plt.ylabel('mAP', fontsize=14)
plt.xticks(index, classes, fontsize=10)
plt.title('RETINANET')
plt.legend(loc='upper left')

autolabel(rects)
plt.savefig(filename)
plt.show()

