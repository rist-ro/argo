import argparse
import numpy as np

def unpack_split(arr):
    word=arr[0]
    parameters = np.array(arr[1:], dtype=np.float)
    l=len(parameters)
    if not l%2==0:
        raise ValueError("Words parameters lenght is %d, not even. The input file passed is expected to contain: u_vec bias_u v_vec bias_v."%l)
    lh=l/2
    u_w = parameters[:lh-1]
    bias_u = parameters[lh-1]
    v_w = parameters[lh:-1]
    bias_v = parameters[-1]
    return (word,u_w,bias_u,v_w,bias_v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', help='The file where to find the parameters of the GloVe model. Each line: word u_vec u_bias v_vec v_bias')
    args = parser.parse_args()
    
    with open(args.inputfile, 'r') as fin:
        words, u_embeddings, u_biases, v_embeddings, v_biases = zip(*[unpack_split(line.rstrip().split(' ')) for line in fin.readlines()])
        u_embeddings=np.array(u_embeddings)
        u_biases=np.array(u_biases)
        v_embeddings=np.array(v_embeddings)
        v_biases=np.array(v_biases)
    
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}
    
    vectors=u_embeddings
    
    evaluate_vectors(vectors, vocab, ivocab)

def evaluate_vectors(W, vocab, ivocab):
    """Evaluate the trained word vectors on a variety of tasks"""
    
    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
    prefix = './eval/question-data/'
    
    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100
    
    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words
    
    for i in xrange(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]
        
        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T
        
        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        
        # the norm is slow,
        # we can use the fact that || ua - ub -uc +ud ||^2 = ||ua - ub - uc||^2 + ||u_d||^2 - 2 (u_a-u_b-u_c) u_d
        # distnorms = [np.linalg.norm((W[a, :] - W[b, :] - W[c, :]) + W, axis=1) for (a,b,c) in zip(ind1,ind2,ind3)]
        d_norms = (np.sum(W ** 2, 1) ** (0.5))
        distnorms = [np.linalg.norm(W[a, :] - W[b, :] - W[c, :]) + d_norms -2*np.dot((W[b, :] - W[a, :]+  W[c, :]), W.T) for (a,b,c) in zip(ind1,ind2,ind3)]
        
        for k,(a,b,c) in enumerate(zip(ind1,ind2,ind3)):
            distnorms[k][a] = np.Inf
            distnorms[k][b] = np.Inf
            distnorms[k][c] = np.Inf
        
        predictionsnorm = np.argmin(distnorms, axis=1)
        
        # scalar product, is valid only if vectors are normalized
        # distprods = [-np.dot((W[b, :] - W[a, :]+  W[c, :]), W.T) for (a,b,c) in zip(ind1,ind2,ind3)]
        # predictionsprod = np.argmin(distprods, axis=1)
        # print(all(predictionsnorm==predictionsprod))
        
        val = (ind4 == predictionsnorm) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("%s:" % filenames[i])
        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
            (np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Semantic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('Syntactic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


if __name__ == "__main__":
    main()
