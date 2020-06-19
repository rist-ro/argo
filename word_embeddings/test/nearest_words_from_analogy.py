"""

Testing of a word embedding, answers to queries a:b=c:d.
It is possible to specify a, b and d; so to obtain the word embedding nearer to a-b+d, i.e. the best approximation for c.

"""

import argparse
import numpy as np
import os, sys
from utils import *

parser = argparse.ArgumentParser(description='Testing of a word embedding, answers to queries a:b=c:d. It is possible to specify a, b and d; so to obtain the word embedding nearer to a-b+d, i.e. the best approximation for c.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('inputfile', help='The file where to find the parameters of the GloVe model. Each line: word u_vec u_bias v_vec v_bias')
parser.add_argument('--howmany', '-hm', type=int, help='Returns the nearest words in the word_embedding space. How many words to return with associated distances.', default=10)
parser.add_argument('--amonghowmany', '-ahm', type=int, help='How many words of the dictionary to consider as candidates, i.e. among how many words to consider the possible "nearest words" to the analogy. -1 for all of them.', default=-1)
parser.add_argument('--vecsize', '-v', required=True, type=int, help='Vector size of the word embedding.')
parser.add_argument('--worda', '-a', help='The word a.', default='king')
parser.add_argument('--wordb', '-b', help='The word b.', default='man')
parser.add_argument('--wordd', '-d', help='The word d.', default='woman')
parser.add_argument('--onlyu', action='store_true', help='If this flag is set, only the u vector is expected, thus only the distance in the euclidean space is possible.')
parser.add_argument('--numthreads', '-nt', type=int, help='Number of threads to use.', default=20)
parser.add_argument('--tolerance', action='store_true', help='This is the tolerance flag in the query. If true, the words of the query are removed from the possible answers. a:b=c:? can never return a, b or c.')

args = parser.parse_args()
vecsize = args.vecsize
onlyu = args.onlyu
tolerance=args.tolerance
numthreads=args.numthreads
os.environ['OMP_NUM_THREADS'] = str(numthreads)
os.environ['NUMEXPR_NUM_THREADS'] = str(numthreads)

dictionary_size, dictionary, reversed_dictionary, u_embeddings, \
    u_biases, v_embeddings, v_biases = \
    read_embeddings(args.inputfile, args.vecsize, onlyu)

if (not onlyu) and (u_embeddings.shape != v_embeddings.shape):
    raise ValueError("The dimensions of the loaded u_embeddings (%d,%d) are not matching with the dimensions of the loaded v_embeddings (%d,%d). Quitting.."%(u_embeddings.shape[0], u_embeddings.shape[1],v_embeddings.shape[0],v_embeddings.shape[1]))

if u_embeddings.shape[0] != len(dictionary):
    raise ValueError("The number of vectors present in the embeddings is %d while the lenght of the dictionary is %d.  Quitting.."%(u_embeddings.shape[0], len(dictionary)))

howmany=args.howmany
amonghowmany=args.amonghowmany
if amonghowmany==-1:
    amonghowmany=None

# a:b=c:d
wa=args.worda
wb=args.wordb
wd=args.wordd

import time

print(" Query result for %s - %s + %s = ?"%(wa,wb,wd))

print("\n u embedding in Euclidean space measures")

start = time.time()
distcalc=DistanceCalculatorMeasuresEuclidean(dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany, amonghowmany, tolerance=tolerance)
iam = distcalc.analogy_query_c(wa, wb, wd)
end = time.time()
print_array([(reversed_dictionary[i], dist) for (i,dist) in iam])
print("time: %f"%(end - start))
print("\n")

print("\n u embedding in Euclidean space target")

start = time.time()
distcalc=DistanceCalculatorTargetEuclidean(dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany, amonghowmany, tolerance=tolerance)
iam = distcalc.analogy_query_c(wa, wb, wd)
end = time.time()
print_array([(reversed_dictionary[i], dist) for (i,dist) in iam])
print("time: %f"%(end - start))
print("\n")

print("\n u embedding on the Sphere, comparison in 0")

start = time.time()
distcalc=DistanceCalculatorMeasuresSpherein0(dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany, amonghowmany, tolerance=tolerance)
iam = distcalc.analogy_query_c(wa, wb, wd)
end = time.time()
print_array([(reversed_dictionary[i], dist) for (i,dist) in iam])
print("time: %f"%(end - start))
print("\n")

print("\n u embedding on the Sphere, comparison in a")

start = time.time()
distcalc=DistanceCalculatorMeasuresSphereinA(dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany, amonghowmany, tolerance=tolerance)
iam = distcalc.analogy_query_c(wa, wb, wd)
end = time.time()
print_array([(reversed_dictionary[i], dist) for (i,dist) in iam])
print("time: %f"%(end - start))
print("\n")

print("\n u embedding on the Sphere, follow logmap and find closest")
start = time.time()
distcalc=DistanceCalculatorTargetSphere(dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany, amonghowmany, tolerance=tolerance)
iam = distcalc.analogy_query_c(wa, wb, wd)
end = time.time()
print_array([(reversed_dictionary[i], dist) for (i,dist) in iam])
print("time: %f"%(end - start))
print("\n")


# print("v embedding")
# relationship_query(wa, wb, wd, v_embeddings)
