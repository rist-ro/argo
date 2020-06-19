
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np
import pdb

task_name = ['comp.sys',
             'rec.sport',
             'sci',
             'religion'
             ]

task_range = [(0.65,0.87),(0.75,0.97),(0.75,1),(0.7,0.90)]

def load_x_y(file_path):
    print(file_path)
    with open(file_path) as f:
        data = f.read()

    data = data.split('\n')
    data = data[:-1]

    x = [float(row.split("\t")[0]) for row in data]
    y = [float(row.split("\t")[1]) for row in data]

    return x, y 
    
#pdb.set_trace()


dirPlots = "plots"

for task in range(len(task_name)):

    dirName = "/home/luigi/word_embedding/20newsgroups/" + task_name[task] + "/"
    #"glove-pu-r0-alpha-accuracy.txt"

    SMALL_SIZE = 16
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE)

    fig = plt.figure(figsize=(10,7))

    file_path = dirName + "glove-p0-r0-alpha-accuracy.txt"
    x, y  = load_x_y(file_path)
    plt.plot(x,y, 'b-', label=r'$\alpha$-embedding')

    file_path = dirName + "glove_pretrained-p0-r0-accuracy.txt"
    x, y  = load_x_y(file_path)
    plt.plot(x,y, 'k-', label=r'glove.6B.300d')

    file_path = dirName + "glove_if_idf-n1-p0-r0-accuracy.txt"
    x, y  = load_x_y(file_path)
    plt.plot(x,y, 'b--', label=r'if_idf')

    #file_path = dirName + "glove_if_idf-n0-p0-r0-accuracy.txt"
    #x, y  = load_x_y(file_path)
    #plt.plot(x,y, 'xb--', label=r'if_idf')

    file_path = dirName + "glove_u-p0-r0-accuracy.txt"
    x, y  = load_x_y(file_path)
    plt.plot(x,y, 'r--', label=r'glove u')


    file_path = dirName + "glove_u+v-p0-r0-accuracy.txt"
    x, y  = load_x_y(file_path)
    plt.plot(x,y, 'g-', label=r'glove u+v')

    plt.xlabel(r'$\alpha$')
    plt.title(task_name[task])

    plt.grid()
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.xlim((-10,5))
    plt.ylim(task_range[task])

    plt.xticks(np.arange(-10, 5, 2))

    
    file_path = dirPlots + "/" + task_name[task] + "_rebuttal.png"
    print(file_path)
    plt.savefig(file_path)



