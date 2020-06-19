from sklearn.datasets import fetch_20newsgroups
import numpy as np

def read_20newsgroup(task=None, ratio_datasets=[0.6, 0.2, 0.2], random_state=42, data_home="/ssd_data/text/"):

    assert np.equal(np.sum(ratio_datasets), 1)
    newsgroups = fetch_20newsgroups(data_home=data_home, subset='all', remove=('headers', 'footers', 'quotes'), categories=task, random_state = random_state)

    #newsgroups_validation_to_be_split = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=task, random_state = seeds[index])

    # filter empty data, it can happen when you remove headers footers and quotes
    all_data, all_targets = zip(*[(d,t) for d,t in zip(newsgroups.data, newsgroups.target) if not d==''])
    all_data = np.array(all_data)
    all_targets = np.array(all_targets)

    n_points = all_targets.shape[0]
    n_point_train = int(n_points * ratio_datasets[0])
    n_point_validation = int(n_points * (ratio_datasets[0]+ratio_datasets[1]))

    train_data = all_data[:n_point_train]
    validation_data = all_data[n_point_train:n_point_validation]
    test_data = all_data[n_point_validation:]

    train_target = all_targets[:n_point_train]
    validation_target = all_targets[n_point_train:n_point_validation]
    test_target = all_targets[n_point_validation:]

    return train_data, train_target, validation_data, validation_target, test_data, test_target

