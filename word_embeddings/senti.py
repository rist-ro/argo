import sklearn as sk

import pandas as pd

import datetime

import numpy as np
import pdb

from pprint import pprint

from utils import save_test_file, save_file
from utils import accuracy_glove, accuracy_tf_idf, load_embedding, load_pretrained_glove
 
# OLD EMBEDDINGS
#dir_alpha_embeddings = '/data/thor_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
#dir_alpha_embeddings = '/data1/text/word_embeddings/enwiki-alpha-emb/enwiki-v300-n1000/'


# NEW EMBEDDINGS
dir_alpha_embeddings = {}
dir_alpha_embeddings['0'] = '/data1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
dir_alpha_embeddings['u'] =  '/data/wonderwoman_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
alphas = np.arange(-3, 3, 0.1)

# ENLARGED EMBEDDINGS
dir_alpha_embeddings = {}
dir_alpha_embeddings['0'] = '/data/captamerica_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
dir_alpha_embeddings['u'] = '/data/captamerica_hd1/text/similarities/NIPS2019-rebattle-2/enwiki-alpha-emb/enwiki-v300-n1000/'
alphas = np.concatenate([np.arange(-10,-3,0.5),np.arange(3,10,0.5)])

dirSenti = 'senti_dataset/'

fileName_alpha = 'alpha[ALPHA]-embeddings-[P0]-enwiki-v300-n1000.txt'
vec_size_alpha = 300
runs = 2

np.random.seed(42)
seeds = np.random.permutation(list(range(runs)))

reference_p = ['0', 'u']
  
plot = 0
#alphas = np.concatenate([np.arange(-10,-3,0.5),np.arange(-3, 3, 0.1), np.arange(3,10,0.5)])
stats = 0

#############################################


if plot==0 and stats==0:
    D_glove, V_glove, D_pretrained_glove, V_pretrained_glove = load_embedding()


# https://gitlab.com/praj88/deepsentiment/blob/master/train_code/utility_functions.py
# Combine and split the data into train and test
def read_senti(path):
    # read dictionary into df
    df_data_sentence = pd.read_table(path + 'dictionary.txt')
    df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split('|', expand=True)
    df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})

    # read sentiment labels into df
    df_data_sentiment = pd.read_table(path + 'sentiment_labels.txt')
    df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
    df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})

                
    #combine data frames containing sentence and sentiment
    df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')
                
    return df_processed_all

            
for r in range(1,runs):
    for p0 in reference_p:
    
        print("run", r, "p", p0)
            
        if stats==1:

            dirName = 'senti/senti/' + 'glove_if_idf-n1-p' + p0 + '-r' + str(r)
            save_test_file(dirName)                
            dirName = 'senti/senti/' + 'glove_if_idf-n0-p' + p0 + '-r' + str(r)
            save_test_file(dirName)
            dirName = 'senti/senti/' + 'glove_u+v-p' + p0 + '-r' + str(r) 
            save_test_file(dirName)
            dirName = 'senti/senti/' + 'glove_u-p' + p0 + '-r' + str(r) 
            save_test_file(dirName)
            dirName = 'senti/senti/' + 'glove_pretrained-p' + p0 + '-r' + str(r)
            save_test_file(dirName)
                
                
            # plot alpha
            accuracies = []
            for alpha in alphas:
                
                dirName = 'senti/senti/' + 'glove-p' + p0 + '-alpha' + "{:.1f}".format(alpha) + '-r' + str(r)
                print(dirName)
                    
                file_path = dirName + "/accuracy_test.txt"
                with open(file_path) as f:
                    data = f.read()

                data = data.split('\n')[0]
                accuracy = float(data.split('\t')[0])
                l2 = float(data.split('\t')[1])   

                accuracies.append([alpha, accuracy, l2])

            dirName = 'senti/senti/' + 'glove-p' + p0 + '-r' + str(r)
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

            # vectorize
            

            
            print("\n\nTraining... ")

            ##########################
        
            accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_tf_idf(train_data,
                                                                    validation_to_be_split_data,
                                                                    train_target,
                                                                    validation_to_be_split_target,
                                                                    run = r)

            print("IF-IDF n1 accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

            dirName = 'senti/senti/' + 'glove_if_idf-n1-p' + p0 + '-r' + str(r) 
            save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
                
            ##########################
        
            accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_tf_idf(train_data,
                                                                                            validation_to_be_split_data,
                                                                                            train_target,
                                                                                            validation_to_be_split_target,
                                                                                            normalizer = False,
                                                                                            run = r)

            print("IF-IDF n0 accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

            dirName = 'senti/senti/' + 'glove_if_idf-n0-p' + p0 + '-r' + str(r) 
            save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
                
            ##########################
        
            accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(train_data,
                                                                   validation_to_be_split_data,
                                                                   train_target,
                                                                   validation_to_be_split_target,
                                                                   D_glove,
                                                                   V_glove,
                                                                   u_plus_v = True,
                                                                   run = r)
    
            print("GLOVE u+v accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

            dirName = 'senti/senti/' + 'glove_u+v-p' + p0 + '-r' + str(r) 
            save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
                
            ##########################
            
            accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(train_data,
                                                                   validation_to_be_split_data,
                                                                   train_target,
                                                                   validation_to_be_split_target,
                                                                   D_glove,
                                                                   V_glove,
                                                                   u_plus_v = False,
                                                                   run = r)
    
            print("GLOVE u accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))

            dirName = 'senti/senti/' + 'glove_u-p' + p0 + '-r' + str(r) 
            save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)

            ##########################
        
            accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(train_data,
                                                                                           validation_to_be_split_data,
                                                                                           train_target,
                                                                                           validation_to_be_split_target,
                                                                                           D_pretrained_glove,
                                                                                           V_pretrained_glove,
                                                                                           u_plus_v = False,
                                                                                           run = r)
            
            print("GLOVE-pretrained u accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test))
                
            dirName = 'senti/senti/' + 'glove_pretrained-p' + p0 + '-r' + str(r) 
            save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
            
            ##########################
            
            for alpha in alphas:
                print("{:.1f}".format(alpha))


                print("Loading glove alpha " + "{:.1f}".format(alpha) + "...")
                start = datetime.datetime.now()        
                path = dir_alpha_embeddings[p0] + fileName_alpha.replace('[ALPHA]', "{:.1f}".format(alpha)).replace('[P0]', p0)
                D_alpha_glove, V_alpha_glove = load_pretrained_glove(path = path,
                                                                     vec_size = vec_size_alpha)                               
                end = datetime.datetime.now()
                delta = end - start
                print("load_pretrained_glove " + str(delta.total_seconds()) + "s")
                
 
                accuracies_train, accuracies_validation, cs, acc_test, c_star = accuracy_glove(train_data,
                                                                       validation_to_be_split_data,
                                                                       train_target,
                                                                       validation_to_be_split_target,
                                                                       D_alpha_glove,
                                                                       V_alpha_glove,
                                                                       u_plus_v = False,
                                                                       max_iter = 100,
                                                                       run = r)
            
                print("GLOVE-alpha" + "{:.1f}".format(alpha) + " accuracy on \ntrain: " + str(["{:.3f}".format(acc_train) for acc_train in accuracies_train]) + "\nvalidation: " + str(["{:.3f}".format(acc_validation) for acc_validation in accuracies_validation]) + "\ntest: " + "{:.3f}".format(acc_test) + " " + "{:.3f}".format(c_star))

                dirName = 'senti/senti/' + 'glove-p' + p0 + '-alpha' + "{:.1f}".format(alpha) + '-r' + str(r) 
                save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star)
            
