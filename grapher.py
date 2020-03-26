import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from io import StringIO
import sys

def autolabel(rects, size=24):
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{0:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', size=size)

matplotlib.rcParams.update({'font.size': 24})
plt.figure(figsize=(18, 10))

r1 = sys.argv[1]
r2 = sys.argv[2] if len(sys.argv)==4 else None
filename = sys.argv[2] if len(sys.argv)==3 else sys.argv[3]

df = pd.read_csv(r1, sep=" ", header=None, names=np.arange(10))

classes = df.iloc[:-3,4].values
index = np.arange(len(classes))

percs = np.array(df.iloc[:-3,8].values,dtype='float32')

if(len(sys.argv)==3):
    rects = plt.bar(index, percs, 0.4, label='RETINANET')
    plt.xlabel('category')
    plt.ylabel('mAP',)
    plt.xticks(index, classes,fontsize=18)
    plt.title('RETINANET')
    # plt.legend(loc='upper left')
    autolabel(rects)
    plt.savefig(filename)
    plt.show()
else:
    df2 = pd.read_csv(r2, sep=" ", header=None, names=np.arange(10))
    percs2 = np.array(df2.iloc[:-3,8].values,dtype='float32')

    rects = plt.bar(index - 0.2, percs, 0.4, label='reduced-classes')
    rects2 = plt.bar(index + 0.2, percs2, 0.4, label='all-classes')
    plt.xlabel('category')
    plt.ylabel('mAP',)
    plt.xticks(index, classes,fontsize=18)
    plt.title('RETINANET')
    plt.legend(loc='upper center')

    autolabel(rects,18)
    autolabel(rects2,18)
    plt.savefig(filename)
    plt.show()

