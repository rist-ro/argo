import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np
import pdb

corpora = ["wikigiga5", "commoncrawl42B", "commoncrawl840B"]
corpora_ls = ['-','--', ":"]

reference_p = ['0']
we_normalization = [0, 1, 2]
normalization = [0, 1, 2]

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

dirName = "/home/luigi/word_embedding/senti/senti/"
#"glove-p0-r0-alpha-accuracy.txt"

SMALL_SIZE = 16
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


for p in reference_p:
    for wen in we_normalization:
        for n in normalization:
    
            fig = plt.figure(figsize=(20,10))


            file_path = dirName + "glove-p" + p + '-wen' + str(wen) + '-n' + str(n) + "-r0-alpha-accuracy.txt"
            x, y  = load_x_y(file_path)
            plt.plot(x,y, 'b-', label=r'$\alpha$-embedding')

            for i, c in enumerate(corpora):
                file_path = dirName + "glove_pretrained_" + c + "-p" + p + '-wen' + str(wen) + '-n' + str(n) + "-r0-accuracy.txt"
                x, y  = load_x_y(file_path)
                plt.plot(x,y, 'k-', label=c, linestyle=corpora_ls[i])

            if n!=1:
                file_path = dirName + "glove_if_idf-p" + p + '-wen' + str(wen) + '-n' + str(n) + "-r0-accuracy.txt"
                x, y  = load_x_y(file_path)
                plt.plot(x,y, 'b--', label=r'if_idf')

            #file_path = dirName + "glove_if_idf-n0-p0-r0-accuracy.txt"
            #x, y  = load_x_y(file_path)
            #plt.plot(x,y, 'xb--', label=r'if_idf')

            file_path = dirName + "glove_u-p" + p + '-wen' + str(wen) + '-n' + str(n) + "-r0-accuracy.txt"
            x, y  = load_x_y(file_path)
            plt.plot(x,y, 'r--', label=r'glove u')

            file_path = dirName + "glove_u+v-p" + p + '-wen' + str(wen) + '-n' + str(n) + "-r0-accuracy.txt"
            x, y  = load_x_y(file_path)
            plt.plot(x,y, 'g-', label=r'glove u+v')

            plt.xlabel(r'$\alpha$')
            plt.title('Stanford Sentiment Treebank p' + p + '-wen' + str(wen) + '-n' + str(n))

            plt.grid()
            plt.legend(loc='upper left')

            plt.tight_layout()
            plt.xlim((-10,10))
            plt.ylim((0.65,0.8))

            #plt.xticks(np.arange(-10, 5, 2))

            file_path = dirPlots + "/senti_rebuttal-p" + p + '-wen' + str(wen) + '-n' + str(n) + ".png"
            print(file_path)
            plt.savefig(file_path)



