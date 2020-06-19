from functools import partial

from collections import OrderedDict


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import os

import datetime
import time

import numpy as np

import pdb

import multiprocessing

from riccardo.read_glove import load_glove, load_pretrained_glove

def save_test_file(dirName):
    os.makedirs(dirName, exist_ok=True)
    fileName = dirName + "-accuracy.txt"
    print(fileName)
    file_path = dirName + "/accuracy_test.txt"
    with open(file_path) as f:
        data = f.read()
    data = data.split('\n')[0]
    accuracy = float(data.split('\t')[0])
    l2 = float(data.split('\t')[1])
    np.savetxt(fileName, np.array([[-10, accuracy, l2],[10, accuracy, l2]]), fmt='%.4f', delimiter='\t')
    print(fileName)


def save_file(dirName, accuracies_train, accuracies_validation, cs, acc_test, c_star):
    data_matrix = np.hstack([np.array(cs).reshape([-1,1]), np.array(accuracies_train).reshape([-1,1]), np.array(accuracies_validation).reshape([-1,1])])
    os.makedirs(dirName, exist_ok=True) 
    np.savetxt(dirName + "/accuracy.txt",
               data_matrix,
               fmt='%.4f',
               delimiter='\t')
    
    data_matrix = np.array([[acc_test, c_star]])
    os.makedirs(dirName, exist_ok=True) 
    np.savetxt(dirName + "/accuracy_test.txt",
               data_matrix,
               fmt='%.4f',
               delimiter='\t')

    
def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2)

def solve(train_X, train_y, validation_X, validation_y, max_iter, args):

    c = args

    print("Start training") 
    model = LogisticRegression(solver='liblinear', #'lbfgs',
                               random_state=42,
                               max_iter = max_iter,
                               C = c)

    start = datetime.datetime.now()
    model.fit(train_X, train_y)
    end = datetime.datetime.now()
    delta = end - start
    print("Training for " + "{:.1f}".format(c) + " took " + str(delta.total_seconds()) + "s")
    
    predicted_classes = model.predict(train_X)
    acc_train = accuracy_score(train_y, predicted_classes)

    predicted_classes = model.predict(validation_X)
    acc_validation = accuracy_score(validation_y, predicted_classes)
        
    global_accuracies_train[c] = acc_train
    global_accuracies_validation[c] = acc_validation
    global_models[c] = model
        
def logistic_regression(train_X, validation_X, test_X, train_y, validation_y, test_y, max_iter = 100, standard_scaler = True, normalizer = False):

    assert(logical_xor(standard_scaler,normalizer) or (standard_scaler==False and normalizer==False))
    
    #accuracies_train = []
    #accuracies_validation = []

    if standard_scaler:
        sc = StandardScaler()
        train_X = sc.fit_transform(train_X)
        validation_X = sc.transform(validation_X)
        test_X = sc.transform(test_X)
    if normalizer:
        sc = Normalizer()
        train_X = sc.fit_transform(train_X)
        validation_X = sc.transform(validation_X)
        test_X = sc.transform(test_X)
        
    cs = np.concatenate([np.arange(0.0001, 0.001, 0.0001),
                         np.arange(0.001, 0.01, 0.001),
                         np.arange(0.01, 0.1, 0.01),
                         np.arange(0.1, 2, 0.2),
                         np.arange(2, 10, ),
                         np.arange(10, 100, 20),
                         np.arange(100, 1000, 200),
                         ######np.arange(1000, 10000, 2000)
                        ])

    manager = multiprocessing.Manager()
    accuracies_train = manager.dict()
    accuracies_validation = manager.dict()
    models =  manager.dict()

    # I need to share the lock globally in all propcesses
    # if I don't share D and V in this way I lose the paralellism of the threads. Whhy? (Luigi)
    def init(accuracies_train, accuracies_validation, models):
        #global lock
        global global_accuracies_train
        global global_accuracies_validation
        global global_models
        global_accuracies_train = accuracies_train
        global_accuracies_validation = accuracies_validation
        global_models = models
        
    pool = multiprocessing.Pool(processes=48,
                                initializer=init,
                                initargs=(accuracies_train, accuracies_validation, models) # lock, 
                               )

    # pass extra parameters to the function
    func = partial(solve, train_X, train_y, validation_X, validation_y, max_iter)
    pool.map(func, cs)
    pool.close()
    pool.join()
    
    
    #accuracies_train = np.array(accuracies_train)
    #accuracies_validation = np.array(accuracies_validation)
    
    #c_star, _ =  max(dict(accuracies_validation).items())
    c_star = max(dict(accuracies_validation), key=lambda k: dict(accuracies_validation)[k])
    
    print("Start retraining") 
    model = LogisticRegression(solver='liblinear', #'lbfgs',
                               random_state=42,
                               max_iter = max_iter,
                               C = c_star)

    #start = datetime.datetime.now()
    #model.fit(train_X, train_y)
    #end = datetime.datetime.now()
    #delta = end - start
    #print("Training for " + "{:.1f}".format(c_star) + " took " + str(delta.total_seconds()) + "s")

    predicted_classes = models[c_star].predict(test_X)
    accuracy_test = accuracy_score(test_y, predicted_classes)

    #parameters = model.coef_

    list_accuracies_train = [j for i,j in sorted(accuracies_train.items(), key=lambda t: t[0])]
    list_accuracies_validation = [j for i,j in sorted(accuracies_validation.items(), key=lambda t: t[0])]
    
    return list_accuracies_train, list_accuracies_validation, cs, accuracy_test, c_star


# see https://scikit-learn.org/stable/modules/preprocessing.html
# see https://stats.stackexchange.com/questions/290958/logistic-regression-and-scaling-of-features
def accuracy_tf_idf(newsgroups_train_data, newsgroups_validation_data, newsgroups_train_target, newsgroups_validation_target, max_iter = 100, standard_scaler = False, normalizer = True, run = 0, n_points_eval = -1):
    
    # train tf_idf
    # see https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest

    vectorizer = TfidfVectorizer()

    train_X = vectorizer.fit_transform(newsgroups_train_data)
    validation_test_X = vectorizer.transform(newsgroups_validation_data)
    if n_points_eval==-1:
        n_points = int(validation_test_X.shape[0]/2)
    else:
        n_points = n_points_eval
    validation_X = validation_test_X[:n_points]
    test_X = validation_test_X[n_points:]
    
    train_y = newsgroups_train_target
    validation_y = newsgroups_validation_target[:n_points]
    test_y = newsgroups_validation_target[n_points:]    
    
    return logistic_regression(train_X, validation_X, test_X, train_y, validation_y, test_y, max_iter = max_iter, standard_scaler = standard_scaler, normalizer = normalizer)


def accuracy_glove(newsgroups_train_data, newsgroups_validation_data, newsgroups_train_target, newsgroups_validation_target, D, V, u_plus_v = False, max_iter = 100, standard_scaler = True, normalizer = False, run = 0, n_points_eval = -1):

    start = datetime.datetime.now()
    print("Start vectorizer_glove... ")
    train_X, train_y = vectorizer_glove(newsgroups_train_data, newsgroups_train_target, D, V, u_plus_v)
    validation_test_X, validation_test_y = vectorizer_glove(newsgroups_validation_data, newsgroups_validation_target, D, V, u_plus_v)

    
    end = datetime.datetime.now()
    delta = end - start
    print("vectorizer_glove " + str(delta.total_seconds()) + "s")
        
    if n_points_eval==-1:
        n_points = int(validation_test_X.shape[0]/2)
    else:
        n_points = n_points_eval
        
    validation_X = validation_test_X[:n_points]
    test_X = validation_test_X[n_points:]
    
    validation_y = validation_test_y[:n_points]
    test_y = validation_test_y[n_points:]    
    
    return logistic_regression(train_X, validation_X, test_X, train_y, validation_y, test_y, max_iter = max_iter, standard_scaler = standard_scaler, normalizer = normalizer)
    
'''
def vectorize3(args):
    #print(i)

    #d, y = args
    print(args)
    time.sleep(5)

    
def vectorize2(D, V, u_plus_v, X, Y, list_documents, labels, args):
    #print(i)

    #d, y = args
    print(args)
    time.sleep(5)
'''
    
def vectorize(u_plus_v, batch_size, list_documents, labels, args):
    #print(i)

    #d, y = args
    #print(args)
    start_index = args

    vectorizer = TfidfVectorizer()
    analyze = vectorizer.build_analyzer()
            
    start = datetime.datetime.now()
    for d, y in zip(list_documents[batch_size*start_index:batch_size*(start_index+1)],
                    labels[batch_size*start_index:batch_size*(start_index+1)]):
        
        document = []
        i = 0
        for w in analyze(d):

            try:
                if u_plus_v:
                    glove = (global_V['u'][global_D[w]] + global_V['v'][global_D[w]])/2
                else:
                    glove = global_V['u'][global_D[w]]
                document.append(glove)
            except KeyError as e:
                i = i+1

            #if i>0:
            #    print("missing words " + str(i) + " / " + str(len(d)))

            #if len(mean_d == 0):
            #    raise Exception("Empty mean_d")

        if len(document)>0:
            mean_d = np.mean(document, axis=0)
            #if X is None:
            #    X = mean_d
            #else:
            #    #pdb.set_trace()
            #    X = np.vstack([X, mean_d])
        
            #with lock:
            global_X.append(mean_d)
            global_Y.append(y)

    end = datetime.datetime.now()
    delta = end - start
    print("process in " + str(delta.total_seconds()) + "s")

        
def vectorizer_glove(list_documents, labels, D, V, u_plus_v = False):

    # see https://blog.ruanbekker.com/blog/2019/02/19/sharing-global-variables-in-python-using-multiprocessing/
    manager = multiprocessing.Manager()
    X = manager.list()
    Y = manager.list()
    
    # see https://scikit-learn.org/stable/modules/feature_extraction.html
    #vectorizer = TfidfVectorizer()
    #analyze = vectorizer.build_analyzer()

    n_elements = len(list_documents)

    # see https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes
    
    # crate a lock
    #lock = multiprocessing.Lock()

    # I need to share the lock globally in all propcesses
    # if I don't share D and V in this way I lose the paralellism of the threads. Whhy? (Luigi)
    def init(D, V, X, Y):
        #global lock
        global global_D
        global global_V
        global global_X
        global global_Y
        #lock = l
        global_D = D
        global_V = V
        global_X = X
        global_Y = Y

    pool = multiprocessing.Pool(processes=24,
                                initializer=init,
                                initargs=(D, V, X, Y) # lock, 
                               )

    batch_size = 5000
    
    # pass extra parameters to the function
    func = partial(vectorize, u_plus_v, batch_size, list_documents, labels)
    pool.map(func, np.arange(0, n_elements, batch_size))
    pool.close()
    pool.join()
    
    return np.array(X), np.array(Y)


def old_vectorizer_glove(list_documents, labels, D, V, u_plus_v = False):
    X = None
    Y = []
    # see https://scikit-learn.org/stable/modules/feature_extraction.html
    vectorizer = TfidfVectorizer()
    analyze = vectorizer.build_analyzer()

    j = 0
    for d, y in zip(list_documents, labels):
        
        document = []
        i = 0
        for w in analyze(d):
            try:
                if u_plus_v:
                    glove = (V['u'][D[w]] + V['v'][D[w]])/2
                else:
                    glove = V['u'][D[w]]
                document.append(glove)
            except KeyError as e:
                i = i+1

        #if i>0:
        #    print("missing words " + str(i) + " / " + str(len(d)))

        #if len(mean_d == 0):
        #    raise Exception("Empty mean_d")

        if len(document)>0:
            mean_d = np.mean(document, axis=0)
            if X is None:
                X = mean_d
            else:
                #pdb.set_trace()
                X = np.vstack([X, mean_d])
            Y.append(y)

        j = j+1
        if j % 1000 == 0:
            print(j)
        
    return X, np.array(Y)

def load_embedding():
        
    print("Loading glove...")
    D_glove, V_glove = load_glove("enwiki", 300, 1000)

    print("Loading pretrained glove...")
    D_pretrained_glove, V_pretrained_glove = load_pretrained_glove("wikigiga5")

    D_pretrained_glove = D_glove
    V_pretrained_glove = V_glove
    return D_glove, V_glove, D_pretrained_glove, V_pretrained_glove
