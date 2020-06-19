import numpy as np

try:
    import pdb
except:
    pass

def load(params):

    # strip from parameters unsupported features
    supported_options = ('dataset_type','n_features','n_points','extrema','seed','custom_x','custom_y')
    fun_params = dict((k, params[k]) for k in supported_options if k in params)

    # Generate the dataset:
    x,y = generate_dataset(**fun_params)


    percentage_train =  0.7 if 'percentage_train' not in params else params['percentage_train']
    n_points = len(y)
    dase = {"n_samples_train": int(percentage_train * n_points),
          "n_samples_test": n_points - int(percentage_train * n_points),
          "train_set_x": x[:int(percentage_train * n_points)],
          "train_set_y": y[:int(percentage_train * n_points)],
          "test_set_x": x[int(percentage_train * n_points):],
          "test_set_y": y[int(percentage_train * n_points):],
          "input_size": x.shape[1],
          "output_size": y.shape[1] if len(y.shape)>1 else 1,
          "binary": 0}

    return dase




def generate_dataset(
        dataset_type='sin_sym',
        n_features=2, n_dep_variables=1,
        n_points=1000,
        extrema=(0,1),
        seed=-1,
        custom_x=None,custom_y=None):
    """
    generate a dataset for regression of arbitrary dimension.
    Linear, polinomial and exponential or generate tge dataset dictionary from custom values

    :param dataset_type:    choose the tipe of function::
                            'white_noise' - dependent variable normally distributed
                            'linear'      - a linear plane of the form :math:`y = A x + b`, A and B are filled with ones
                            'sin_sym'     - a very non linear sinusoidal shape :math:`y = b \sin (x^2)`
                            'custom_xy'   - create a dataset from custom_x and custom_y
    :param n_features:      dimension of the input space
    :param n_dep_variables: dimension of the output space
    :param n_points:        number of points to be generated
    :param extrema:         the sampling space of the dependent variables.
                            if (min,max) it describes the coordinates of the vertices of the (hyper-)cube
                            if ((min_1,max_1),...,(min_n_features,max_n_features)) set the coordinates of the (hyper-)parallelepiped
    :param seed:            set the seed

    :param custom_x         generate dataset once provided x
    :param custom_y         generate dataset once provided y

    :return:                an array of feature data and an array of the dependent variables


    TODO: Possible improvments, add parameters to the functios

    """

    if seed != -1:
        np.random.seed(seed)

    x_points = sample_features(n_features, n_points, extrema, seed)
    if dataset_type == 'white_noise':
        y_points = np.random.normal(size=(n_points,n_dep_variables))

    elif dataset_type == 'linear':
        A = np.ones(shape=(x_points.shape[1],n_dep_variables))
        b = np.ones(shape=(n_dep_variables))
        y_points = (np.matmul(x_points,A)+b)

    elif dataset_type == 'sin_sym':
        b = np.ones(shape=(n_dep_variables))
        y_points = np.sin(np.sum(x_points**2,axis=1))

    elif dataset_type == 'custom_xy':
        assert(custom_x.shape[0]==len(y)),'input and target do not match'
        x_points=custom_x
        y_points=custom_y


    return x_points, y_points


def sample_features(
        n_features=2,
        n_points=1000,
        extrema=(0,1),
        seed=-1):
    """
    generate samples of the feature space or x space

    :param n_features:      dimension of the input space
    :param n_dep_variables: dimension of the output space
    :param n_points:        number of points to be generated
    :param extrema:         the sampling space of the dependent variables.
                            if (min,max) it describes the coordinates of the vertices of the (hyper-)cube
                            if ((min_1,max_1),...,(min_n_features,max_n_features)) set the coordinates of the (hyper-)parallelepiped
    :param seed:            set the seed

    :return:                an array of x points
                            rows are single points in the x space
    """
    if seed!=-1:
        np.random.seed(seed)


    extrema_np = np.array(extrema)
    if extrema_np.shape == (2,):
        # extrema with form (min,max)
        min_x, max_x = extrema

        # every row represents a datapoint
        x_points = np.random.uniform(low=min_x, high=max_x, size=(n_points, n_features))
    else:
        # extrema with form ((min_1,max_1),...,(min_n_features,max_n_features))
        assert (extrema_np.shape[1] == 2), 'extrema with wrong shape'
        if n_features !=2 :
            assert (n_features == extrema_np.shape[0]), 'extrema is not compatible with n_features'

        n_features = extrema_np.shape[0]

        x_points = np.random.uniform(low=extrema_np[:,0], high=extrema_np[:,1], size=(n_points, n_features))


    return x_points



