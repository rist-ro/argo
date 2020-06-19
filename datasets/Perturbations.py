import tensorflow as tf

import numpy as np

import pdb

def context_encoding(image, size):
    #if isinstance(images, list):
    #    return tuple([image for image in images])
    #else:
    #pdb.set_trace()

    '''
    matrix = np.zeros([28, 28, 1])
    matrix[0:14, 0:14, :] = 1
    mask = tf.convert_to_tensor(matrix, dtype=tf.int32)

    indexes = tf.where(tf.equal(mask, 1))
    '''

    #indices = tf.constant([[0, 0, 0], [0, 1, 0] [1, 0, 0], [1, 1, 0]])

    w, h, c = image.get_shape().as_list()

    if c!=1:
        raise Exception("contextual encoding not implemented for color images, feel free to implement it")

    x = np.random.randint(w-size)
    y = np.random.randint(h-size)
    
    rows, columns = np.meshgrid(range(x,x+size), range(y,y+size))
        
    channel = np.zeros(size**2, dtype=np.int32)
    indices = tf.stack([[int(r) for r in rows.flatten()], [int(c) for c in columns.flatten()], channel], axis=1)

    updates = np.ones(size**2, dtype=np.int32)*-1 # black square
    ce_image = tf.tensor_scatter_nd_update(image, indices, updates)
    
    return ce_image   
    
