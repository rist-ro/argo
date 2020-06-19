import argparse, os, operator, itertools
import numpy as np
np.set_printoptions(precision=6, threshold=np.inf)
from numpy import linalg
import readers, spaces
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import Counter
from kneed import KneeLocator
from scipy import interpolate
from scipy.signal import argrelextrema
import progressbar
import numexpr as ne
from mylogging import *
from spaces import EuclideanSpace, HyperSphere


parser = argparse.ArgumentParser(description='Extract word embeddings stats.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser()

parser.add_argument('--vecsize', '-v', required=True, type=int, help='Vector size of the word embedding.')
parser.add_argument('inputfile', nargs='+', help='The files where to find the parameters of the GloVe model. Each line: word u_vec u_bias v_vec v_bias.')
parser.add_argument('--outputfolder', '-o', required=True, help='folder where to save output datafiles')
parser.add_argument('--vocabulary', action='store_true', help='If this flag is specified the vocabulary will be determined from the first inputfile and will be used for all the files.')
# parser.add_argument('--labels', type=str, nargs='+', default=None, help='Labels for the input files.')

group = parser.add_mutually_exclusive_group()
group.add_argument('--uplusvhalf', action='store_true', help='If this flag is set, only the first vectors will be considered for each word (if there are other numbers after vecsize they will be ignored). This vector will be interpreted as (u+v)/2. Only the distances in the euclidean space are possible.')
group.add_argument('--uandv', action='store_true', help='If this flag is set, both u vectors and v vectors will be considered separately for the pca and for the norms in the euclidean space.')

parser.add_argument('--onsimplex', action='store_true', help='If this flag is set, will calculate also statistics on the simplex (and probability distributions).')
parser.add_argument('--wordsfile', help='File from which to take the words to consider to output their probabilities.')



def get_pca(matrix, ncomponents):
    # OLD HAND-MADE PCA
    # # _, pca_eigenvalues, _ = np.linalg.svd(matrix)
    # # rowvar is False means rows contain observations
    # cov_matrix = np.cov(matrix, rowvar=False)
    # # cov_matrix = self.covariance_matrix(matrix)
    # pca_eigenvalues, _ = np.linalg.eig(cov_matrix)
    # # sort eigevals in descending order
    # pca_eigs_index = np.argsort(-pca_eigenvalues)
    # pca_eigenvalues = pca_eigenvalues[pca_eigs_index]
    # return pca_eigenvalues
    #
    # SKLEARN PCA
    pca=PCA(n_components=ncomponents).fit(matrix)
    return pca.explained_variance_


def analyze_embeddings(embeddings, x_0, space, embeddings_label, outputbasename):
    """Analyze the embeddings distribution in the space and save results to files.
    
    Args:
        embeddings (list of numpy.array): embeddings to analyze.
        x_0 (numpy.array): The uniform distribution, used as a reference for the space we are analyzing.
        space (class Space): the space to consider (has methods median, dist, etc...).
        embeddings_label (string): label to use for saving the data (used for the future plots).
        outputbasename (string): base name for the output file.
    """
    
    x_bar = space.mean(embeddings)
    
    avg_sqrtmsd, avg_dmin, avg_dmax, avg_dmean, avg_dstd, avg_dhi = space.dist_hist(embeddings, x_bar, bins='auto')
    ys, xs = avg_dhi
    save_data([xs[:-1]], [ys], [embeddings_label+"-from-avg"], outputbasename+"-from-avg-hist.dat",
            plot_method_name="bar", kwargs_list=[{'align':'edge', 'width':np.diff(xs)}]
            )
    
    un_sqrtmsd, un_dmin, un_dmax, un_dmean, un_dstd, un_dhi = space.dist_hist(embeddings, x_0, bins='auto')
    ys, xs = un_dhi
    save_data([xs[:-1]], [ys], [embeddings_label+"-from-un"], outputbasename+"-from-un-hist.dat",
            plot_method_name="bar", kwargs_list=[{'align':'edge', 'width':np.diff(xs)}]
            )
    
    dist_avg_un = space.dist(np.array([x_bar]), x_0)[0]
    
    with open(outputbasename+'-summary.txt','w') as outstream:
        outstream.write("STATS ON THE EMBEDDINGS, obtained in the space: %s\n"%space.__class__.__name__)
        outstream.write("-----------------------------------------------------------\n\n")
        outstream.write("uniform (x_0)\n")
        outstream.write(str(x_0))
        outstream.write("\n")
        outstream.write("mean (x_bar)\n")
        outstream.write(str(x_bar))
        outstream.write("\n")
        rows=[("dist(x_bar, x_0)", dist_avg_un)]
        write_rows_to_txt(rows, outstream, spacer='  ')
        
        outstream.write("\n-----------------------------------------------------------\n\n")
        outstream.write("distances from x_0:\n")
        rows = [("sqrtmeansqdist","min","max","mean","std"), (un_sqrtmsd, un_dmin, un_dmax, un_dmean, un_dstd)]
        write_rows_to_txt(rows, outstream, spacer='  ')
        
        outstream.write("\n-----------------------------------------------------------\n\n")
        outstream.write("distances from x_bar:\n")
        rows = [("sqrtmeansqdist","min","max","mean","std"), (avg_sqrtmsd, avg_dmin, avg_dmax, avg_dmean, avg_dstd)]
        write_rows_to_txt(rows, outstream, spacer='  ')


def get_distribution_stats(mu_embeddings, words, selected_words, outputdistribbasename):
    #get elbow for all words
    #1 how many words before the elbow in average
    #2 10 words with max n of words before the elbow
    #3 10 words with min n of words before the elbow
    #4 distances from the center of learned words
    
    #save distribution for selected words
    ntuples=[]
    # cnt=0
    for i in progressbar.progressbar(range(len(words))):
        word=words[i]
        # i=dictionary[word]
        
        # cnt+=1
        # if cnt==30:
        #     break
        
        mu_embedding=mu_embeddings[i]
        indexes=np.argsort(-mu_embedding)
        ordered_mu_embedding = mu_embedding[indexes]
        ordered_words = np.array(words)[indexes]
        
        ntuple = index_before_elbow(word, ordered_mu_embedding)
        ntuples.append(ntuple)
        
        if word in selected_words:
            #save the distribution
            thiswordname = outputdistribbasename+'-'+word.replace('/','_')
            save_distribution(ordered_mu_embedding, ordered_words, inputlabel+'-'+word, thiswordname+'.dat')
            n=ntuple[-1]+10
            save_txt([ordered_words[:n],ordered_mu_embedding[:n]], thiswordname+'-before_elbow.txt')
    
    # import pdb;pdb.set_trace()
    ntuples = np.array(ntuples)
    nmeans = np.mean(ntuples, axis=0)
    nstds = np.sqrt(np.mean(np.power(ntuples-nmeans,2), axis=0))
    
    #SORT based on keys 5,2,1 and get the arg
    to_sort = [(tup[1][5], tup[1][2], tup[1][1], tup[0]) for tup in enumerate(ntuples)]
    ordidxs = np.array(list(zip(*sorted(to_sort)))[-1])
    words = np.array(words)
    
    # WRITE STATS TO FILE
    #1 how many words before the elbow in average
    #2 10 words with max n of words before the elbow
    #3 10 words with min n of words before the elbow
    #4 distances from the center of learned words
    
    with open(outputdistribbasename+'-distribution-summary.txt','w') as outstream:
        outstream.write("STATS ON THE LEARNED CONDITIONAL DISTRIBUTIONS OF THE WORDS\n")
        outstream.write("-----------------------------------------------------------\n\n")
        rows = [("method","mean","std")]+list(zip(methodname,nmeans,nstds))
        write_rows_to_txt(rows, outstream, spacer='  ')
        
        outstream.write("\n-----------------------------------------------------------\n\n")
        outstream.write("10 words with min n of words before the elbow:\n")
        header = ["word"]+methodname
        first_words=words[ordidxs[:10]]
        first_tuples=ntuples[ordidxs[:10]]
        rows = [header]+[list(itertools.chain([w],t)) for (w,t) in zip(first_words,first_tuples)]
        write_rows_to_txt(rows, outstream, spacer='  ')
        
        outstream.write("\n-----------------------------------------------------------\n\n")
        outstream.write("10 words with max n of words before the elbow:\n")
        last_words=words[ordidxs[-10:]]
        last_tuples=ntuples[ordidxs[-10:]]
        rows = [header]+[list(itertools.chain([w],t)) for (w,t) in zip(last_words,last_tuples)]
        write_rows_to_txt(rows, outstream, spacer='  ')
    

def write_distribs_from_selected(mu_embeddings, words, selected_words, dictionary, outputdistribbasename):
    #Save distribution for selected words, in case I do not compute all stats.
    #it is better to have it separated from the loop in get_stats
    #since maybe I do not want to recompute all the stats
    
    for i in progressbar.progressbar(range(len(selected_words))):
        selword = selected_words[i]
        selidx = dictionary[selword]
        mu_embedding = mu_embeddings[selidx]
        
        indexes = np.argsort(-mu_embedding)
        ordered_mu_embedding = mu_embedding[indexes]
        ordered_words = np.array(words)[indexes]
        ntuple = index_before_elbow(selword, ordered_mu_embedding)
        #save the distribution
        thiswordname = outputdistribbasename+'-'+selword.replace('/','_')
        save_distribution(ordered_mu_embedding, ordered_words, inputlabel+'-'+word, thiswordname+'.dat')
        n=ntuple[-1]+10
        save_txt([ordered_words[:n],ordered_mu_embedding[:n]], thiswordname+'-before_elbow.txt')


color=['k','r','g','y','m','c']
methodname=['kneedle', 'frac-kneedle', '2nd-deriv', '2-mean', 'diff-frac', 'diff-frac-value-low']

def plot_with_demarcation(word, indexes, ordered_prob, ntuple, xlim=None, ylim=None,):
    plt.title(word)
    plt.plot(indexes, ordered_prob, 'bx-')
    plt.xlim(xlim)
    plt.ylim(ylim)
    for i,n in enumerate(ntuple):
        plt.vlines(n, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color=color[i], label=methodname[i])
    
    plt.legend()
    plt.show()

def __normalize(a):
    return (a - min(a)) / (max(a) - min(a))

# def elbow_find(x,y):
#
#     if not np.array_equal(np.array(x), np.sort(x)):
#         raise ValueError('x values must be sorted')
#
#     xn = __normalize(x)
#     yn = __normalize(y)
#
#     # theta = np.radians(45)
#     # c, s = np.cos(theta), np.sin(theta)
#     # R = np.array(((c,-s), (s, c)))
#
#     # import pdb;pdb.set_trace()
#     idxs = np.array([0,-1])
#     linen = interpolate.interp1d(xn[idxs], yn[idxs])
#     ylinen = linen(xn)
#     yd = np.array(ylinen)-yn-xn
#     idmax = np.argmax(yd)
#
#     from matplotlib import pyplot as plt
#     plt.plot(xn,ylinen, 'x-')
#     plt.plot(xn,yn,'x-')
#     plt.plot(xn,yd, 'o-')
#
#     plt.vlines(xn[idmax], plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
#     plt.show()
#
#     # # import pdb;pdb.set_trace()
#     # halflen=int(np.ceil(len(xn)/2.))
#     # idxs = np.array([0,halflen])
#     # linenhalf = interpolate.interp1d(xn[idxs], yn[idxs])
#     # xnhalf = xn[:halflen]
#     # ylinenhalf = linenhalf(xnhalf)
#
#     # xnrot,ynrot = zip(*np.matmul(list(zip(xn,yn)),R.T))
#     # xnrot=np.array(xnrot)
#     # ynrot=np.array(ynrot)
#     #
#     # idxs = np.array([0,-1])
#     # line = interpolate.interp1d(xnrot[idxs], ynrot[idxs])
#     # yline = line(xnrot)
#
#     # yd = np.array(yline)-np.array(ynrot)
#     # idmax = np.argmax(yd)
#
#     # from matplotlib import pyplot as plt
#     # plt.plot(xnrot,yline, 'x-')
#     # plt.plot(xnrot,ynrot,'x-')
#     # plt.plot(xnrot,yd, 'o-')
#     #
#     # plt.vlines(xnrot[idmax], plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
#     #
#     # plt.show()
#
#     import pdb;pdb.set_trace()
#     # Calculate difference curve
#     # print("finding an elbow in a decreasing function")
#     # yd = ylinen - yn
#     return idmax

from scipy.optimize import curve_fit

def func(x, a, b):
    return a*x+b

def invf(x, a0, alpha):
    return a0*np.power(x,alpha)

def expofit_elbow(x,y):
    # x=np.arange(700, dtype=float)
    # y=np.exp(-x)+0.01*np.random.normal(size=x.shape)
    # y=abs(y)*1.1

    mni=1
    mxi=300

    # xn=__normalize(x)[mni:mxi]
    # yn=__normalize(y)[mni:mxi]
    xn=x[mni:mxi]
    yn=np.log(y[mni:mxi])
    
    # initial_guess_exp=(1,-1)
    # weights=np.power(x[mni:mxi],-5)
    popt, pcov = curve_fit(func, xn, yn)#, sigma=weights, p0=initial_guess_exp)
    
    a,b=popt
    
    # from matplotlib import pyplot as plt
    # plt.plot(xn,yn,'o')
    # plt.plot(xn,weights,'--')
    # plt.plot(xn,expf(*[xn]+list(initial_guess_exp)),'-', label='initialguess')
    # plt.plot(xn,expf(xn,a0,alpha),'-',label='fitted')
    # # plt.plot(xn[1:],invf(xn[1:],a0i,alphai),'-')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(x,y,'o')
    # plt.plot(xn,np.exp(expf(xn,a0,alpha)),'-',label='fitted')
    # plt.show()
    #
    tau = int(np.ceil(-1/a))
    # plt.vlines(xnrot[idmax], plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    
    return tau

def std_check(arr, bound):
    return np.std(arr)<bound

def std_and_value_check(arr, bound):
    return (np.std(arr)<bound and arr[0]<bound)

def elbow_from_diffs(x,y,low_trigger=False):
    y=np.asarray(y)
    s=10
    frac=10.
    refd=y[0]/frac
    # import pdb;pdb.set_trace()
    # diffs=y[:-s]-y[s:]
    # index=next(i for i in range(len(diffs)-1) if diffs[i]<refd)+s
    condition=std_check
    if low_trigger:
        condition=std_and_value_check
    
    index=next((i for i in range(len(y)) if condition(y[i:i+s], refd)), len(y))
    return index


def index_before_elbow(word, ordered_prob):
    """Find the index of the knee for an ordered probability distribution, expected ordered.

    Args:
        ordered_prob (type): the ordered probability distribution.

    Returns:
        type: a tuple with the different indexes calculated by different methods (e.g. kneedle, 2nd derivative, 2-means and hierarchical clustering single linkage selecting 2 clusters, ...).

    """
    
    # #TIME
    # import time
    # t=[]
    # t.append(time.time())
    # #ENDTIME
    
    # Kneedle algorithm https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf
    # code: https://github.com/arvkevi/kneed
    dict_size = len(ordered_prob)
    indexes = np.arange(dict_size)
    # n_kneedle = elbow_simple(indexes, ordered_prob)
    kn = KneeLocator(indexes, ordered_prob, direction='decreasing', invert=True, label=word)
    #it is possible that the KneeLocator does not find any local maxima, in this case the probability is approx equal for all words
    n_kneedle = kn.knee if kn.knee else dict_size
    
    # #TIME
    # t.append(time.time())
    # #ENDTIME
    
    # Simple Kneedle
    # indexes = np.arange(len(ordered_prob))
    # n_simplekneedle = elbow_find(indexes, ordered_prob)
    n_simplekneedle = int(np.ceil(n_kneedle*(2/3)))
    
    # #TIME
    # t.append(time.time())
    # #ENDTIME
    
    #Max second derivative
    ysn = __normalize(ordered_prob)
    secondDerivatives = np.array([(ysn[i+1] + ysn[i-1] - 2 * ysn[i]) for i in indexes[1:-1]])
    n_snddv = np.argmax(secondDerivatives)+1
    
    # #TIME
    # t.append(time.time())
    # #ENDTIME
    
    #Clustering 2-mean
    kmeans = KMeans(n_clusters=2).fit(ordered_prob.reshape(-1,1))#, init=np.array([0,1]).reshape(-1,1)).fit(ordered_prob.reshape(-1,1))
    target = kmeans.labels_[0]
    compl = 1 if target==0 else 0
    i0k=next(i for (i,l) in enumerate(kmeans.labels_) if l==compl)
    counter1k=Counter(kmeans.labels_)[target]
    if not i0k==counter1k:
        raise ValueError("found anomaly in data k")
    
    # #TIME
    # t.append(time.time())
    # #ENDTIME
    
    #DIFFERENCE FRACTION
    n_diff = elbow_from_diffs(indexes, ordered_prob)
    
    # #TIME
    # t.append(time.time())
    # #ENDTIME
    
    #DIFFERENCE FRACTION
    n_diffnfrac = elbow_from_diffs(indexes, ordered_prob, low_trigger=True)
    
    # #TIME
    # t.append(time.time())
    # #ENDTIME
    #
    # #Agglomerative clustering simple linkage, looking for 2 clusters
    # agglom = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average').fit(ordered_prob.reshape(-1,1))
    # target=1
    # compl=0
    # i0a=next(i for (i,l) in enumerate(agglom.labels_) if l==compl)
    # if i0a==0:
    #     target=0
    #     compl=1
    #     i0a=next(i for (i,l) in enumerate(agglom.labels_) if l==compl)
    # counter1a=Counter(agglom.labels_)[target]
    # if not i0a==counter1a:
    #     raise ValueError("found anomaly in data a")
    #
    # #TIME
    # t.append(time.time())
    # #ENDTIME
    
    # #Fraction of the maximum probability
    # plim=ordered_prob[0]/3.
    # n_fracmax=next(i for i,p in enumerate(ordered_prob) if p<plim)
    #
    
    # #TIME
    # t.append(time.time())
    # t=np.array(t)
    # deltas=t[1:]-t[:-1]
    # print("times:  "+"  ".join(map(str,deltas)))
    # #ENDTIME
    ntuple=(n_kneedle, n_simplekneedle, n_snddv, i0k, n_diff, n_diffnfrac) #, i0a
    
    # print(ntuple)
    # plot_with_demarcation(word, indexes, ordered_prob, ntuple, xlim=(-5,1000))
    
    return ntuple


def make_folders(inputlabel):
    outputdatafolder=outputfolder+'/'+inputlabel
    outputpcafolder=outputdatafolder+'/pca'
    outputdistsfolder=outputdatafolder+'/dists'
    outputdistribfolder=outputdatafolder+'/words_distributions'
    os.makedirs(outputpcafolder, exist_ok=True)
    os.makedirs(outputdistsfolder, exist_ok=True)
    os.makedirs(outputdistribfolder, exist_ok=True)
    return (outputpcafolder, outputdistsfolder, outputdistribfolder)


args = parser.parse_args()

if args.onsimplex:
    if not args.wordsfile:
        raise ValueError("if onsimplex flag is set I will need a file with the words to consider to output their probabilities (Just a simple one column text file with words).")
vecsize=args.vecsize
consideronlyfirstvec=args.uplusvhalf

outputfolder=args.outputfolder
if outputfolder.endswith('/'):
  outputfolder=outputfolder[:-1]

inputlabels=list(map(readers.rmtxt, map(os.path.basename,args.inputfile)))

# if not len(inputlabels)==len(args.inputfile):
#     raise ValueError("Number of labels must match number of inputfiles.")

embeddings_list=[]
labels=[]
if args.onsimplex:
    embeddings_list_simplex=[]

bool_vocabulary=args.vocabulary

words_set=None
if bool_vocabulary:
    reader=readers.get_reader(args.inputfile[0])
    (dictionary_size, dictionary, reversed_dictionary) = \
            reader.read_dictionary(args.inputfile[0])
    words_set=set(dictionary.keys())

for inputname,inputlabel in zip(args.inputfile,inputlabels):
    print("processing %s\n"%inputname)
    reader=readers.get_reader(inputname)
    dictionary_size, dictionary, reversed_dictionary, u_embeddings, v_embeddings = \
        reader.read_embeddings(inputname, vecsize, consideronlyfirstvec, words_set=words_set)
    
    outputbasename = readers.rmtxt(os.path.basename(inputname))
    outputpcafolder, outputdistsfolder, outputdistribfolder = make_folders(inputlabel)
    
    outputdistsbasename=outputdistsfolder+'/'+outputbasename
    outputpcabasename=outputpcafolder+'/'+outputbasename
    outputdistribbasename=outputdistribfolder+'/'+outputbasename
    # uniform distributions
    eucl = EuclideanSpace()
    u_0 = np.zeros(u_embeddings.shape[1])
    hyps = HyperSphere()
    x_0 = np.ones(dictionary_size)/np.sqrt(dictionary_size)
    
    if args.uplusvhalf:
        #if vectors are given as (u+v)/2
        analyze_embeddings(u_embeddings, u_0, eucl, inputlabel+'-upvh', outputdistsbasename+'-dists-uplusvhalf')
        save_pca([u_embeddings], [inputlabel+'-upvh'], outputpcabasename+'-pca-uplusvhalf.dat')
    elif args.uandv:
        #if both u and v vectors are given
        analyze_embeddings(u_embeddings, u_0, eucl, inputlabel+'-u', outputdistsbasename+'-dists-u')
        analyze_embeddings(v_embeddings, u_0, eucl, inputlabel+'-v', outputdistsbasename+'-dists-v')
        save_pca([u_embeddings], [inputlabel+'-u'], outputpcabasename+'-pca-u.dat')
        save_pca([v_embeddings], [inputlabel+'-v'], outputpcabasename+'-pca-v.dat')
    else:
        #if only u vectors are given
        analyze_embeddings(u_embeddings, u_0, eucl, inputlabel+'-u', outputdistsbasename+'-dists-u')
        save_pca([u_embeddings], [inputlabel+'-u'], outputpcabasename+'-pca-u.dat')
    
    if args.onsimplex:
        if not args.uplusvhalf:
            mu_embeddings = spaces.calculate_mu_embeddings(u_embeddings, v_embeddings)
            x_embeddings = ne.evaluate('sqrt(mu_embeddings)')
            
            analyze_embeddings(x_embeddings, x_0, hyps, inputlabel+'-x', outputdistsbasename+'-dists-x-on-hyps')
            
            # TODO pca on the hypersphere
            save_pca([mu_embeddings], [inputlabel+'-mu'], outputpcabasename+'-pca-on-simplex.dat')
            
            selected_words=readers.read_selected_words(args.wordsfile)
            words=[reversed_dictionary[i] for i in range(len(reversed_dictionary))]
            get_distribution_stats(mu_embeddings, words, selected_words, outputdistribbasename)
