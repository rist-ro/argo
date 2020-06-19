import logging
import pickle
import sys
import numpy as np
import pandas as pd

def init_logger(logname):
    logging.basicConfig(format='%(message)s', filename=logname, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)

    # # create file handler which logs even debug messages
    fh = logging.FileHandler(logname, mode='w')
    # create console handler
    # ch = logging.StreamHandler(sys.stdout)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    # logger.addHandler(ch)
    
    return logger

def init_stream_logger():
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # # create file handler which logs even debug messages
    # fh = logging.FileHandler(logname)
    # create console handler
    return logger

def read_prob_from_wordcounts_csv(vocab_name):
    df = pd.read_csv(vocab_name, names=["words", "counts"], sep=" ")
    df["probs"] = df["counts"]/df["counts"].sum()
    df.set_index("words", inplace=True)
    return df["probs"]

# the save data format is a dictionary = { plot_method: .., x_label: .. , y_label: .. , data: .., etc..}
def save_data(xs_list, ys_list, label_list, x_label, y_label, outname, plot_method_name="plot"):
    # if kwargs_list:
    #     #TODO use args and kwargs, store label in kwargs
    #     data=[{'plot_method': plot_method_name, 'xs': xs, 'ys': ys, 'label':label, 'kwargs':kwargs} \
    #         for (xs,ys,label,kwargs) in zip(xs_list,ys_list,label_list,kwargs_list)]
    # else:
    
    dict_tosave = create_plot_data_dict(xs_list, ys_list, label_list, x_label, y_label, plot_method_name)
    
    with open(outname,'wb') as outstream:
        pickle.dump(dict_tosave, outstream)
    
    return dict_tosave

    
def save_pca(embeddings_list, labels, outname, n_components=300):
    eigs_list = [get_pca(embeddings, n_components) for embeddings in embeddings_list]
    eigenvals_idx = np.arange(n_components)+1
    
    save_data(itertools.cycle([eigenvals_idx]), eigs_list, labels, outname)

# def save_norms(embeddings_list, labels, outname):
#
#     norms_list = [np.linalg.norm(embeddings, axis=1) for embeddings in embeddings_list]
#
#     save_data(norms_list, itertools.cycle(['hist']), labels, outname)

def save_distribution(mu_embedding, words, label, outname):
    save_data([np.arange(len(words))+1], [mu_embedding], [label], outname)

def strconv(obj):
    if type(obj) in [float, np.float, np.float32, np.float64]:
        return '%.6f'%obj
    else:
        return str(obj)

def write_columns_to_txt(list_of_list_of_stuffs, outstream, spacer=' '):
    for arr in zip(*list_of_list_of_stuffs):
        line = spacer.join([strconv(n) for n in arr])
        line += '\n'
        outstream.write(line)

def write_rows_to_txt(list_of_list_of_stuffs, outstream, spacer=' '):
    for arr in list_of_list_of_stuffs:
        line = spacer.join([strconv(n) for n in arr])
        line += '\n'
        outstream.write(line)

def save_txt(list_of_list_of_stuffs, outname, mode='w'):
    with open(outname,mode) as outstream:
        write_columns_to_txt(list_of_list_of_stuffs, outstream)
