from sklearn.datasets import fetch_20newsgroups

#from sklearn.metrics import accuracy_score

import os

import numpy as np
import pdb

from pprint import pprint

from utils import save_test_file, save_file
from utils import accuracy_glove, accuracy_tf_idf, load_embedding, load_pretrained_glove

'''
categories = {
  'alt.atheism' : 0,
  'comp.graphics' : 1,
  'comp.os.ms-windows.misc' : 2,
  'comp.sys.ibm.pc.hardware' : 3,
  'comp.sys.mac.hardware' : 4,
  'comp.windows.x' : 5,
  'misc.forsale' : 6,
  'rec.autos' : 7,
  'rec.motorcycles' : 8,
  'rec.sport.baseball' : 9,
  'rec.sport.hockey': 10,
  'sci.crypt' : 11,
  'sci.electronics' : 12,
  'sci.med' : 13,
  'sci.space' : 14,
  'soc.religion.christian': 15,
  'talk.politics.guns': 16,
  'talk.politics.mideast': 17,
  'talk.politics.misc': 18,
  'talk.religion.misc': 19
}
'''


binary_classification_tasks = [['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware'],
                               ['rec.sport.baseball','rec.sport.hockey'],
                               ['sci.med','sci.space'],
                               ['alt.atheism','soc.religion.christian']
                               ]


task_name = ['comp.sys',
             'rec.sport',
             'sci',
             'religion'
             ]
 
# OLD EMBEDDINGS
#dir_alpha_embeddings = '/data/thor_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
#dir_alpha_embeddings = '/data1/text/word_embeddings/enwiki-alpha-emb/enwiki-v300-n1000/'


# NEW EMBEDDINGS
dir_alpha_embeddings = {}
dir_alpha_embeddings['0'] = '/data/ironman_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
dir_alpha_embeddings['u'] =  '/data/wonderwoman_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
alphas = np.arange(-3, 3, 0.1)

# ENLARGED EMBEDDINGS
dir_alpha_embeddings = {}
dir_alpha_embeddings['0'] = '/data/captamerica_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
dir_alpha_embeddings['u'] =  '/data/captamerica_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
alphas = np.concatenate([np.arange(-10,-3,0.5),np.arange(3,10,0.5)])

fileName_alpha = 'alpha[ALPHA]-embeddings-[P0]-enwiki-v300-n1000.txt'
vec_size_alpha = 300
runs = 5

np.random.seed(42)
seeds = np.random.permutation(list(range(runs)))

#

reference_p = ['0', 'u']
  
plot = 0
#alphas = np.concatenate([np.arange(-10,-3,0.5),np.arange(-3, 3, 0.1), np.arange(3,10,0.5)])
stats = 0

#############################################

if plot==0 and stats==0:
    D_glove, V_glove,  D_pretrained_glove, V_pretrained_glove = load_embedding()

    
for r in range(runs):
    for p0 in reference_p:
    
        for index, task in enumerate(binary_classification_tasks):

            print("run", r, "p", p0, str(task))
            
            #dirName = '20newsgroups/' + task_name[index]
            #os.makedirs(dirName, exist_ok=True) 
    
            if stats==1:

                dirName = '20newsgroups/' + task_name[index] + '/glove_if_idf-n1-p' + p0 + '-r' + str(r)
                save_test_file(dirName)

                dirName = '20newsgroups/' + task_name[index] + '/glove_if_idf-n0-p' + p0 + '-r' + str(r)
                save_test_file(dirName)
                dirName = '20newsgroups/' + task_name[index] + '/glove_u+v-p' + p0 + '-r' + str(r) 
                save_test_file(dirName)
                dirName = '20newsgroups/' + task_name[index] + '/glove_u-p' + p0 + '-r' + str(r) 
                save_test_file(dirName)
                dirName = '20newsgroups/' + task_name[index] + '/glove_pretrained-p' + p0 + '-r' + str(r)
                save_test_file(dirName)

                
                # plot alpha
                accuracies = []
                for alpha in alphas:

                    dirName = '20newsgroups/' + task_name[index] + '/glove-p' + p0 + '-alpha' + "{:.1f}".format(alpha) + '-r' + str(r)
                    print(dirName)
                    
                    file_path = dirName + "/accuracy_test.txt"
                    with open(file_path) as f:
                        data = f.read()
                        
                    data = data.split('\n')[0]
                    accuracy = float(data.split('\t')[0])
                    
                    accuracies.append([alpha, accuracy])

                dirName = '20newsgroups/' + task_name[index] + '/glove-p' + p0 + '-r' + str(r)
                #os.makedirs(dirName, exist_ok=True)
                fileName = dirName + "-alpha-accuracy.txt"
                print(fileName)
                np.savetxt(fileName,
                           np.array(accuracies),
                           fmt='%.4f',
                           delimiter='\t')

            elif plot==1:

                pass
                
            else:
                     
                newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=task, random_state = seeds[r])
                #newsgroups_validation_to_be_split = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=task, random_state = seeds[index])

                n_points = newsgroups.target.shape[0]
                n_point_train = int(n_points*0.6)
                newsgroups_train_data = newsgroups.data[:n_point_train]
                newsgroups_validation_to_be_split_data = newsgroups.data[n_point_train:]
                newsgroups_train_target = newsgroups.target[:n_point_train]
                newsgroups_validation_to_be_split_target = newsgroups.target[n_point_train:]
            
            
                #print(len(newsgroups_train.data))
                #print(newsgroups_train.target.shape)
                #print(newsgroups_train.filenames.shape)

                
                print("\n\nTraining... " + str(task))

                ##########################
        
                accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_tf_idf(newsgroups_train_data,
                                                                    newsgroups_validation_to_be_split_data,
                                                                    newsgroups_train_target,
                                                                    newsgroups_validation_to_be_split_target,
                                                                    run = r)

                print("IF-IDF n1 accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

                dirName = '20newsgroups/' + task_name[index] + '/glove_if_idf-n1-p' + p0 + '-r' + str(r) 
                save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)

                ##########################
        
                accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_tf_idf(newsgroups_train_data,
                                                                                            newsgroups_validation_to_be_split_data,
                                                                                            newsgroups_train_target,
                                                                                            newsgroups_validation_to_be_split_target,
                                                                                            normalizer = False,
                                                                                            run = r)

                print("IF-IDF n0 accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

                dirName = '20newsgroups/' + task_name[index] + '/glove_if_idf-n0-p' + p0 + '-r' + str(r) 
                save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)

                ##########################
        
                accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(newsgroups_train_data,
                                                                   newsgroups_validation_to_be_split_data,
                                                                   newsgroups_train_target,
                                                                   newsgroups_validation_to_be_split_target,
                                                                   D_glove,
                                                                   V_glove,
                                                                   u_plus_v = True,
                                                                   run = r)
    
                print("GLOVE u+v accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

                dirName = '20newsgroups/' + task_name[index] + '/glove_u+v-p' + p0 + '-r' + str(r) 
                save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
    
                ##########################
        
                accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(newsgroups_train_data,
                                                                   newsgroups_validation_to_be_split_data,
                                                                   newsgroups_train_target,
                                                                   newsgroups_validation_to_be_split_target,
                                                                   D_glove,
                                                                   V_glove,
                                                                   u_plus_v = False,
                                                                   run = r)
    
                print("GLOVE u accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

                dirName = '20newsgroups/' + task_name[index] + '/glove_u-p' + p0 + '-r' + str(r) 
                save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)

                ##########################
        
                accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(newsgroups_train_data,
                                                                   newsgroups_validation_to_be_split_data,
                                                                   newsgroups_train_target,
                                                                   newsgroups_validation_to_be_split_target,
                                                                   D_pretrained_glove,
                                                                   V_pretrained_glove,
                                                                   u_plus_v = False,
                                                                   run = r)
            
                print("GLOVE-pretrained u accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))
                
                dirName = '20newsgroups/' + task_name[index] + '/glove_pretrained-p' + p0 + '-r' + str(r) 
                save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
                
                ##########################
            
                for alpha in alphas:
                    print("{:.1f}".format(alpha))

                    print("Loading glove alpha " + "{:.1f}".format(alpha) + "...")
                    path = dir_alpha_embeddings[p0] + fileName_alpha.replace('[ALPHA]', "{:.1f}".format(alpha)).replace('[P0]', p0)
                    D_alpha_glove, V_alpha_glove = load_pretrained_glove(path = path,
                                                                         vec_size = vec_size_alpha)        
                    accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(newsgroups_train_data,
                                                                       newsgroups_validation_to_be_split_data,
                                                                       newsgroups_train_target,
                                                                       newsgroups_validation_to_be_split_target,
                                                                       D_alpha_glove,
                                                                       V_alpha_glove,
                                                                       u_plus_v = False,
                                                                       max_iter = 100,
                                                                       run = r)
            
                    print("GLOVE-alpha" + "{:.1f}".format(alpha) + " accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

                    dirName = '20newsgroups/' + task_name[index] + '/glove-p' + p0 + '-alpha' + "{:.1f}".format(alpha) + '-r' + str(r) 
                    save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
            
