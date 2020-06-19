import tensorflow as tf

import pdb

def adjust_brightness(images, max_delta):
    delta = tf.random_uniform([1], -max_delta, max_delta, dtype=tf.float32) #shape, min, max
    if isinstance(images, list):
        return tuple([tf.image.adjust_brightness(image, delta) for image in images])
    else:
        return tf.image.adjust_brightness(images, delta)

def adjust_brightnessAsym(images, min_delta, max_delta, mask=0):
    delta = tf.random_uniform([1], min_delta, max_delta, dtype=tf.float32) #shape, min, max

    if mask == 1:

        if isinstance(images, list):

            raise Exception("talk to Luigi")
            return tuple([tf.image.adjust_brightness(image, delta) for image in images])
        else:

            # do not apply the transformation to the mask
            
            init_dims = [0 for s in images.shape[:-1].as_list()]
            end_dims = [-1 for s in images.shape[:-1].as_list()]

            mask = tf.slice(images, init_dims + [0], end_dims + [1])
            image = tf.slice(images, init_dims + [1], end_dims + [-1])

            image = tf.image.adjust_brightness(image, delta)
            
            return tf.concat([mask, image], axis=-1)
    else:
        if isinstance(images, list):
            return tuple([tf.image.adjust_brightness(image, delta) for image in images])
        else:
            return tf.image.adjust_brightness(images, delta)
        
    
def flip_up_down(images):
    if isinstance(images, list):
        return tuple([tf.image.flip_up_down(image) for image in images])
    else:
        return tf.image.flip_up_down(images)

def flip_left_right(images):
    if isinstance(images, list):
        return tuple([tf.image.flip_left_right(image) for image in images])
    else:
        return tf.image.flip_left_right(images)
    
def rot90(images):
    if isinstance(images, list):
        return tuple([tf.image.rot90(image) for image in images])
    else:
        return tf.image.rot90(images)
    
    
