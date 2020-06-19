import sklearn as sk

import datetime

import numpy as np
import pdb

#from pprint import pprint

pdb.set_trace()

from utils import save_test_file, save_file
from utils import accuracy_glove, accuracy_tf_idf, load_embeddings, load_pretrained_glove

from utils import read_senti

from test.core.load_embeddings import load_emb_base, load_embeddings_ldv_hdf, load_glove, load_pretrained_glove, get_alpha_ldv_name, get_limit_ldv_name

from sklearn.preprocessing import StandardScaler, Normalizer

corpora = ["wikigiga5", "commoncrawl42B", "commoncrawl840B"]

# OLD EMBEDDINGS
#dir_alpha_embeddings = '/data/thor_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
#dir_alpha_embeddings = '/data1/text/word_embeddings/enwiki-alpha-emb/enwiki-v300-n1000/'

'''
# NEW EMBEDDINGS
dir_alpha_embeddings = {}
dir_alpha_embeddings['0'] = '/data1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
dir_alpha_embeddings['u'] =  '/data/wonderwoman_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
#alphas = np.arange(-3, 3, 0.1)
alphas = np.arange(-3, 3, 0.2)
'''

'''
# ENLARGED EMBEDDINGS
dir_alpha_embeddings = {}
dir_alpha_embeddings['0'] = '/data/captamerica_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
dir_alpha_embeddings['u'] = '/data/captamerica_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
#alphas = np.concatenate([np.arange(-10,-3,0.5),np.arange(3,10,0.5)])
alphas = np.concatenate([np.arange(-10,-3,0.5),np.arange(3,10,0.5)])
'''

# on ww
embdir = "/data/wonderwoman_hd1/text/alpha-embeddings/enwiki-alpha-emb/enwiki-v300-n1000"
# on ironman
#embdir = "/ssd_data/word_embeddings/from_ww/alpha-embeddings-unnorm-pfact/enwiki-alpha-emb/enwiki-v300-n1000"

print("Loading alphas and matrices")
alphas, I0, Iu = load_emb_base(embdir)

print("Inverting matrices")
I0_inv = np.linalg.inv(I0)
Iu_inv = np.linalg.inv(Iu)

pdb.set_trace()

dirSenti = 'senti_dataset/'
log_dir = 'new_senti/senti/'

runs = 1
start_run = 0

np.random.seed(42)
seeds = np.random.permutation(list(range(runs)))

reference_p = ['0']
we_normalization = ['0', 'SI', 'SF', 'NI', 'NF']
normalization = ['0', 'SI', 'SF', 'NI', 'NF']
  
#alphas = np.concatenate([np.arange(-10,-3,0.5),np.arange(-3, 3, 0.1), np.arange(3,10,0.5)])
stats = 0

#############################################

if stats==0:
    D_glove, V_glove_original, D_pretrained_glove, V_pretrained_glove_original = load_embeddings(corpora)    
            
for r in range(start_run,runs):
    for p0 in reference_p:
        for wen in we_normalization:
            for n in normalization:
            
                print("run", r, "p", p0, "wen", wen, "n", n)

                if stats==1:


                    if n[0]!='S':
                        dirName = log_dir + 'glove_if_idf-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r)
                        save_test_file(dirName)                
                    
                    dirName = log_dir + 'glove_u+v-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r) 
                    save_test_file(dirName)
                    dirName = log_dir + 'glove_u-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r) 
                    save_test_file(dirName)
                    for c in corpora:
                        dirName = log_dir + 'glove_pretrained_' + c + '-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r)
                        save_test_file(dirName)
                
                
                    # plot alpha
                    accuracies = []
                    for alpha in alphas:

                        # numerical trick
                        if abs(alpha) < 0.0001:
                            alpha = 0
                            
                        print("run", r, "p", p0, "wen", wen, "n", n, "alpha", alpha)
                        
                        dirName = log_dir + 'glove-p' + p0 + '-alpha' + "{:.1f}".format(alpha) + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r)
                        print(dirName)
                    
                        file_path = dirName + "/accuracy_test.txt"
                        with open(file_path) as f:
                            data = f.read()

                        data = data.split('\n')[0]
                        accuracy = float(data.split('\t')[0])
                        l2 = float(data.split('\t')[1])   

                        accuracies.append([alpha, accuracy, l2])

                    dirName = log_dir + 'glove-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r)
                    #os.makedirs(dirName, exist_ok=True)
                    fileName = dirName + "-alpha-accuracy.txt"
                    print(fileName)
                    np.savetxt(fileName,
                               np.array(accuracies),
                               fmt='%.4f',
                               delimiter='\t')

                #elif plot==1:
                #
                #    pass
        
                else:

                    '''
                    # read split test and train
                    file_path = dirSenti + '/datasetSplit.txt' 
                    with open(file_path) as f:
                        data = f.read()
                    data = data.split('\n')
                    data = data[1:]
                    data = data[:-1]
            
                    split = np.array([[int(i) for i in row.split(",")] for row in data])
            
                    # read labels
                    file_path = dirSenti + '/sentiment_labels.txt' 
                    with open(file_path) as f:
                        data = f.read()
                    data = data.split('\n')
                    # skip 2 since it starts with the sentence 0
                    data = data[2:]
                    data = data[:-1]
            
                    labels = np.array([[int(row.split("|")[0]), float(row.split("|")[1])] for row in data])
                    labels = labels[:split.shape[0],:]
            
                    # read sentences
                    file_path = dirSenti + '/datasetSentences.txt' 
                    with open(file_path) as f:
                        data = f.read()
                    data = data.split('\n')
                    data = data[1:]
                    data = data[:-1]
            
                    sentences = np.array([[int(row.split("\t")[0]), row.split("\t")[1]] for row in data])

                    train_X = sentences[split[:,1]==1,:][:,1]
                    eval_X = sentences[split[:,1]==3,:][:,1]
                    test_X = sentences[split[:,1]==2,:][:,1]

                    train_y = labels[split[:,1]==1,:][:,1]
                    eval_y = labels[split[:,1]==3,:][:,1]
                    test_y = labels[split[:,1]==2,:][:,1]
                    '''

                    if wen == 1:
                        sc = Normalizer()
                    elif wen == 2:
                        sc = StandardScaler()
                    elif wen !=0 :
                        raise Exception("word embedding normalization '" + str(wen) + "' not supported ")
                
                    if wen == 0:
                        V_glove = V_glove_original
                        V_pretrained_glove = V_pretrained_glove_original
                    else:
                        V_glove = {}
                        V_glove['u'] = sc.fit_transform(V_glove_original['u'])
                        V_glove['v'] = sc.fit_transform(V_glove_original['v'])
                        V_pretrained_glove = {}
                        for c in corpora:
                            V_pretrained_glove[c] = {}
                            V_pretrained_glove[c]['u'] = sc.fit_transform(V_pretrained_glove_original[c]['u'])
                            if V_pretrained_glove_original[c]['v'][0] != None:
                                V_pretrained_glove[c]['v'] = sc.fit_transform(V_pretrained_glove_original[c]['u'])                 
                    
                    if n == 0:
                        standard_scaler = False
                        normalizer = False
                    elif n == 1:
                        standard_scaler = True
                        normalizer = False
                    elif n == 2:
                        standard_scaler = False
                        normalizer = True
                    else:
                        raise Exception("normalization '" + str(n) + "' not supported ")

                    
                    data = read_senti(dirSenti + '/')
                    data = sk.utils.shuffle(data, random_state = seeds[r])
                        
                    n_points = len(data)
                    n_point_train = int(n_points*0.6)
                    train_data = data[:n_point_train]["Phrase"].values
                    validation_to_be_split_data = data[n_point_train:]["Phrase"].values
                    train_target = data[:n_point_train]["sentiment_values"].values
                    validation_to_be_split_target = data[n_point_train:]["sentiment_values"].values
                                
                    # to binary labels
                    train_target = (train_target.astype(float)>0.5).astype(int)
                    validation_to_be_split_target = (validation_to_be_split_target.astype(float)>0.5).astype(int)

                    print("\nTraining... ")
                
                    ########################## TF-IDF

                    # see https://datascience.stackexchange.com/questions/33730/should-i-rescale-tfidf-features
                    if n!=1:
                        accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_tf_idf(
                            train_data,
                            validation_to_be_split_data,
                            train_target,
                            validation_to_be_split_target,
                            standard_scaler = standard_scaler,
                            normalizer = normalizer)
            
                        print("IF-IDF n0 accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))
                
                        dirName = log_dir + 'glove_if_idf-p' + p0 +'-wen' + str(wen) + '-n' + str(n) + '-r' + str(r) 
                        save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
                
                    ########################## glove trained by Riccardo u+v
        
                    accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(
                        train_data,
                        validation_to_be_split_data,
                        train_target,
                        validation_to_be_split_target,
                        D_glove,
                        V_glove,
                        u_plus_v = True,
                        standard_scaler = standard_scaler,
                        normalizer = normalizer)
    
                    print("GLOVE u+v accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

                    dirName = log_dir + 'glove_u+v-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r) 
                    save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
                
                    ########################## glove trained by Riccardo u
            
                    accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(
                        train_data,
                        validation_to_be_split_data,
                        train_target,
                        validation_to_be_split_target,
                        D_glove,
                        V_glove,
                        u_plus_v = False,
                        standard_scaler = standard_scaler,
                        normalizer = normalizer)
    
                    print("GLOVE u accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

                    dirName = log_dir + 'glove_u-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r) 
                    save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
            
                    ########################## glove pretrained
            
                    for c in corpora:
                        accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(
                            train_data,
                            validation_to_be_split_data,
                            train_target,
                            validation_to_be_split_target,
                            D_pretrained_glove[c],
                            V_pretrained_glove[c],
                            u_plus_v = False,
                            standard_scaler = standard_scaler,
                            normalizer = normalizer)
            
                        print("GLOVE-pretrained " + c + " u accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))
                
                        dirName = log_dir + 'glove_pretrained_' + c + '-p' + p0 + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r) 
                        save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
            
                    ##########################
            
                    for alpha in alphas:

                        # numerical trick
                        if abs(alpha) < 0.0001:
                            alpha = 0
                            
                        print("{:.1f}".format(alpha))


                        print("Loading glove alpha " + "{:.1f}".format(alpha) + "...")
                        start = datetime.datetime.now()        
                        path = dir_alpha_embeddings[p0] + fileName_alpha.replace('[ALPHA]', "{:.1f}".format(alpha)).replace('[P0]', p0)
                        D_alpha_glove, V_alpha_glove_original = load_pretrained_glove(path = path,
                                                                                      vec_size = vec_size_alpha)                               
                        end = datetime.datetime.now()
                        delta = end - start
                        print("load_pretrained_glove " + str(delta.total_seconds()) + "s")

                        if wen == 1:
                            sc = Normalizer()
                        elif wen == 2:
                            sc = StandardScaler()
                        elif wen !=0 :
                            raise Exception("word embedding normalization '" + str(wen) + "' not supported ")
                
                        if wen == 0:
                            V_alpha_glove = V_alpha_glove_original
                        else:
                            V_alpha_glove = {}
                            #pdb.set_trace()
                            V_alpha_glove['u'] = sc.fit_transform(V_alpha_glove_original['u'])
                            if V_alpha_glove_original['v'][0] is not None:
                                V_alpha_glove['v'] = sc.fit_transform(V_alpha_glove_original['v'])

                            
                        accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(
                            train_data,
                            validation_to_be_split_data,
                            train_target,
                            validation_to_be_split_target,
                            D_alpha_glove,
                            V_alpha_glove,
                            u_plus_v = False,
                            max_iter = 100,
                            standard_scaler = standard_scaler,
                            normalizer = normalizer)
                    
                        print("GLOVE-alpha" + "{:.1f}".format(alpha) + " accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test) + " " + "{:.3f}".format(c_star))

                        dirName = log_dir + 'glove-p' + p0 + '-alpha' + "{:.1f}".format(alpha) + '-wen' + str(wen) + '-n' + str(n) + '-r' + str(r) 
                        save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
            
