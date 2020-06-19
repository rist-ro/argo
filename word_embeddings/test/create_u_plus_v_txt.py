import argparse
import numpy as np
import utils

#this script get a glove embedding with u u_biases v v_biases and creates a file with (u+v)/2 for comparizon purposes with pretrained gloves

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='The file where to find the parameters of the GloVe model. Each line: word u_vec u_bias v_vec v_bias.')
parser.add_argument('--vecsize', '-v', required=True, type=int, help='Vector size of the word embedding.')
args = parser.parse_args()

vecsize=args.vecsize

with open(args.inputfile, 'r') as fin:
    words, u_embeddings, u_biases, v_embeddings, v_biases = zip(*[utils.unpack_split(line.rstrip().split(' '), vecsize, False) for line in fin.readlines()])
    u_embeddings = np.array(u_embeddings)
    u_biases = np.array(u_biases)
    v_embeddings = np.array(v_embeddings)
    v_biases = np.array(v_biases)

outputfile = utils.rmtxt(args.inputfile)+"-u_plus_v_half.txt"
with open(outputfile, 'w') as fout:
    for word, u_w, v_w in zip(words, u_embeddings, v_embeddings):
        arr=[]
        arr.append(word)
        outvec=(u_w+v_w)/2
        arr = arr + ["%.5f"%o for o in outvec]
        line=' '.join(arr)+"\n"
        fout.write(line)
