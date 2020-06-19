def create_id(regularizer_tuple):
    regularizer_name, regularizer_kwargs = regularizer_tuple

    _id = ''

    if regularizer_name == 'autoencoding_regularizer':
        if regularizer_kwargs["distance_function"] == 'euclidean':
            _id += '_AEe'
        elif regularizer_kwargs["distance_function"] == 'wasserstein':
            _id += '_AEw'
        elif regularizer_kwargs["distance_function"] == 'kl':
            _id += '_AEkl'
        elif regularizer_kwargs["distance_function"] == 'fisher':
            _id += '_AEf'
        else:
            raise Exception("distance_function unknown: " + regularizer_kwargs["distance_function"])
    elif regularizer_name == 'perceptual_loss':
        _id += '_PL'
        
    else:
        raise Exception("regularizer unknown: " + regularizer_name)

    _id += '_s' + str(regularizer_kwargs["scale"])

    return _id
