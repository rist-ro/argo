import pickle
import numpy as np

def save_collection(collection_path, collection):
    # save analogies accuracies
    with open(collection_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(collection, f, pickle.HIGHEST_PROTOCOL)


def load_collection_and_backup(collection_path):
    try:
        with open(collection_path, 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            analogies = pickle.load(f)

        with open(collection_path+".bkp", 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(analogies, f, pickle.HIGHEST_PROTOCOL)

    except:
        analogies = {}

    return analogies


def check_done_in_collection(collection, alphas, name_start, enpd):
    # enpd = expected number of methods per dataset

    expected_number = len(collection)*enpd
    done_alphas = 0
    lengths = []
    limit_count = 0

    # e.g. name_start = "u-plog"
    limit_start = "limit-"+name_start

    for d in collection:
        for m in collection[d]:
            if m.startswith(name_start):
                curve = collection[d][m]
                # check how long are the curves of the non-limit ones
                lengths.append(len(curve))
            elif m.startswith(limit_start):
                length = len(collection[d][m])
                np.testing.assert_equal(length, 1, "limit curve must have lenght one, found {:}...".format(length))
                # count how many limits have been done
                limit_count+=1

    nonlimit_count = len(lengths)

    done_limit = False
    # if I found some limits, they must be all
    if limit_count>0:

        # number of limits should be same of expected_number
        if limit_count!=expected_number:
            raise Exception("expected 0 or {:} limits, found {:} instead..".format(expected_number, limit_count))

        print("found already limits done")
        done_limit = True

    # if I found curves, they must be all same length
    if nonlimit_count>0:
        if not len(set(lengths))==1:
            raise Exception("error reading collection, found different lenghts for alpha")

        # number of nonlimits should be same of expected_number
        if nonlimit_count!=expected_number:
            raise Exception("expected 0 or {:} methods for {:}, found {:} instead..".format(expected_number, name_start, nonlimit_count))

        done_alphas = lengths[0]
        print("found already {:} alphas done: {:}".format(done_alphas, alphas[:done_alphas]))

    return done_alphas, done_limit

