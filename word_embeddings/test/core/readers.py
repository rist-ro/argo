import numpy as np
import operator, os, itertools
from abc import ABC, abstractmethod
import numexpr as ne
ne.set_num_threads(20)

def rmtxt(s):
    if s.endswith(".txt"):
        s=os.path.splitext(s)[0]
    return s

def get_reader(inputfilename):
    basename=os.path.basename(inputfilename)
    reader=None
    if basename.startswith('glove'):
        reader=GloVeEmbeddingsFileReader()
    elif basename.startswith('word2vec'):
        reader=Word2vecEmbeddingsFileReader()
    else:
        raise RuntimeError('the inputfilename \'%s\'does not start with either glove or word2vec so I do not know how to read the word embeddings'%basename)
    
    return reader

def read_selected_words(inputname):
    with open(inputname, 'r') as fin:
        words = [line.rstrip().split()[0] for line in fin.readlines()]
    return words


class EmbeddingsFileReader(ABC):
    
    @abstractmethod
    def preprocess(self, fin):
        """ what to do to the newly opened text file as preprocessing """
        pass
    
    @abstractmethod
    def tuple_from_params(self, parameters):
        pass

    def read_dictionary(self, inputname):
        """ read the dictionary from inputfile """
        with open(inputname, 'r') as fin:
            self.preprocess(fin)
            words = [line.rstrip().split(' ')[0] for line in fin.readlines()]
        return self.dicts_from_wordslist(words)

    def read_word_counts(self, inputname):
        """ read the word counts from inputfile """
        with open(inputname, 'r') as fin:
            self.preprocess(fin)
            counts = [int(line.rstrip().split(' ')[1]) for line in fin.readlines()]
        return counts

    def dicts_from_wordslist(self, words):
        dictionary_size = len(words)
        dictionary = {w: idx for idx, w in enumerate(words)}
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return (dictionary_size, dictionary, reversed_dictionary)
    
    def unpack_split_line(self, line, vecsize, onlyu):
        arr=line.rstrip().split(' ')
        word=arr[0]
        parameters = np.array(arr[1:], dtype=np.float)
        
        return list(itertools.chain([word], self.tuple_from_params(parameters, vecsize, onlyu)))
    
    def read_embeddings(self, inputname, vecsize, consideronlyfirstvec, words_set=None):
        """ read the embeddings from inputfile """
        with open(inputname, 'r') as fin:
            self.preprocess(fin)
            words, u_embeddings, v_embeddings = self.get_embeddings_fromfile(fin, vecsize, consideronlyfirstvec, words_set)
        
        dictionary_size, dictionary, reversed_dictionary = self.dicts_from_wordslist(words)
        
        # u_biases and v_biases are not returned at the moment since we do not know what to do with them
        return (dictionary_size, dictionary, reversed_dictionary, u_embeddings, v_embeddings)


class GloVeEmbeddingsFileReader(EmbeddingsFileReader):
    
    def preprocess(self, fin):
        pass
    
    def tuple_from_params(self, parameters, vecsize, onlyu):
        
        l=len(parameters)
        if l!=vecsize and l!=2*vecsize+2:
            raise ValueError("the vecsize passed is not compatible with the observation of line lenghts in the inputfile: line length = %s"%l)
        
        u_w = parameters[:vecsize]
        if onlyu:
            bias_u = None
            v_w = None
            bias_v = None
        else:
            bias_u = parameters[vecsize]
            v_w = parameters[vecsize+1:-1]
            bias_v = parameters[-1]
        
        return (u_w,bias_u,v_w,bias_v)
    
    def get_embeddings_fromfile(self, filestream, vecsize, consideronlyfirstvec, words_set=None):
        words, u_embeddings, u_biases, v_embeddings, v_biases = \
            zip(*[self.unpack_split_line(line, vecsize, consideronlyfirstvec) \
            for line in filestream.readlines()])
        if words_set:
            words, u_embeddings, u_biases, v_embeddings, v_biases = zip(*[(w,uw,bu,vw,bv) for w,uw,bu,vw,bv in zip(words,u_embeddings,u_biases,v_embeddings,v_biases) if w in words_set])
        u_embeddings = np.array(u_embeddings)
        u_biases = np.array(u_biases)
        v_embeddings = np.array(v_embeddings)
        v_biases = np.array(v_biases)
        return (words, u_embeddings, v_embeddings)
    

class Word2vecEmbeddingsFileReader(EmbeddingsFileReader):
    
    def preprocess(self, fin):
        """ here I need to skip the header and the first word if it is <\s> - (what is this tag that word2vec introduces?) """
        fin.readline()
        word = fin.readline().split(' ')[0]
        if not word=='</s>':
            fin.seek(0)
            fin.readline()
    
    def tuple_from_params(self, parameters, vecsize, onlyu):
        l=len(parameters)
        if l!=vecsize and l!=2*vecsize:
            raise ValueError("the vecsize passed is not compatible with the observation of line lenghts in the inputfile: line length = %s"%l)
        
        u_w = parameters[:vecsize]
        if onlyu:
            v_w = None
        else:
            v_w = parameters[vecsize:]
        
        return (u_w,v_w)
    
    def get_embeddings_fromfile(self, filestream, vecsize, consideronlyfirstvec, words_set=None):
        words, u_embeddings, v_embeddings = zip(*[self.unpack_split_line(line, vecsize, consideronlyfirstvec) for line in filestream.readlines()])
        if words_set:
            words, u_embeddings, v_embeddings = zip(*[(w,uw,vw) for w,uw,vw in zip(words,u_embeddings,v_embeddings) if w in words_set])
        u_embeddings = np.array(u_embeddings)
        v_embeddings = np.array(v_embeddings)
        return (words, u_embeddings, v_embeddings)
    

def extract_vocabulary_from(vocabfile):
    with open(vocabfile, 'r') as fin:
        vocab_words = [line.rstrip().split(' ')[0] for line in fin.readlines()]
        vocab_words=set(vocab_words)

    #
    #
    # def __init__(self, dictionary, howmany=10, amonghowmany=None):
    #     self.dictionary=dictionary
    #     self.dictionary_size=len(dictionary)
    #     self.howmany=howmany
    #     self.amonghowmany=amonghowmany
    #     self.x_0 = np.sqrt(np.ones(self.dictionary_size)/self.dictionary_size)
    #
    # def word_analogy_measures(self, u_a, u_b, u_d, u_embeddings, v_embeddings, space="euclidean"):
    #     """ which vector uc_star in u_embeddings is the one with the highest analogy_measure? """
    #     if space=="euclidean":
    #         analogy_measure=self.analogy_measure_euclidean
    #     elif space=="sphere_in_0":
    #         analogy_measure=self.analogy_measure_on_the_sphere_in0
    #     elif space=="sphere_in_a":
    #         analogy_measure=self.analogy_measure_on_the_sphere_ina
    #     elif space=="sphere_logmap":
    #         #follows the logmaps and query the nearest one
    #         analogy_measure=self.analogy_measure_on_the_sphere_logmap
    #     else:
    #         raise ValueError("Unrecognized space argument in find_closest function. space was %s"%space)
    #
    #     uc_star = sorted([(i,analogy_measure(u_a, u_b, uc, u_d, v_embeddings)) for (i,uc) in enumerate(u_embeddings[:self.amonghowmany])], key=operator.itemgetter(1))[:self.howmany]
    #     return uc_star

    # def analogy_nearby(self, word_a, word_b, word_d, u_embeddings, v_embeddings, space="euclidean"):
    #     """given three words a,b,d I want to find c such that a:b=c:d."""
    #     try:
    #         a=self.dictionary[word_a]
    #         b=self.dictionary[word_b]
    #         d=self.dictionary[word_d]
    #     except KeyError as kerr:
    #         print("\nKey Error: {0}".format(kerr))
    #         print("The word requested is not present in the dictionary.\n")
    #         sys.exit(-1)
    #
    #     u_a, u_b, u_d = u_embeddings[a], u_embeddings[b], u_embeddings[d]
    #
    #     #iam is indexes and analogy_measures ordered by analogy measures. list of (i, measure)
    #     iam = self.word_analogy_measures(u_a, u_b, u_d, u_embeddings, v_embeddings, space)
    #
    #     return iam
    #
    # #DEPRECATED here just for backward compatibility test
    #
    # def analogy_nearby_sphere_closest(self, word_a, word_b, word_d, u_embeddings, v_embeddings):
    #     """given three words a,b,d I want to find c such that a:b=c:d."""
    #     try:
    #         a=self.dictionary[word_a]
    #         b=self.dictionary[word_b]
    #         d=self.dictionary[word_d]
    #     except KeyError as kerr:
    #         print("\nKey Error: {0}".format(kerr))
    #         print("The word requested is not present in the dictionary.\n")
    #         sys.exit(-1)
    #
    #     x_target = self.follow_logmap_on_the_sphere(u_embeddings[a], u_embeddings[b], u_embeddings[d], v_embeddings)
    #     x_embeddings = [send_u_to_x_on_the_sphere(u, v_embeddings) for u in u_embeddings]
    #     ans = self.find_closest_euclidean(x_target, x_embeddings)
    #     return ans
    #
    #

def print_array(arr):
    mw = max(len(w) for w,d in arr)
    for (w,d) in arr:
        print(" "+"\t".join((w.ljust(mw),str(d))))


def write_hdf(x, table_name='embeddings', outputname="table_test.hdf"):
    with tables.open_file(outputname, 'w') as f:
        atom = tables.Atom.from_dtype(x.dtype)

        vec_size = 300
        array_c = f.create_earray(f.root, table_name, atom, (0, vec_size))

        chunk_size = 500
        for i in range(0, 70000, chunk_size):
            f.root.embeddings.append(x[i: i + chunk_size])


def read_hdf(filename, table_name='embeddings'):
    with tables.open_file(filename) as f:
        # print(f.root.embeddings)
        x = f.root[table_name][:, :]

    return x
