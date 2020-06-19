import tensorflow as tf


def euclidean(h1, h2):
    return tf.norm(h1 - h2, axis=1)


def wasserstein(params1, params2):
    mu1, sigma1 = params1
    mu2, sigma2 = params2
    wd = tf.sqrt(tf.square(tf.norm(mu1 - mu2, axis=1)) + tf.square(tf.norm(sigma1 - sigma2, axis=1)))
    return wd


# https://www.sciencedirect.com/science/article/pii/S0166218X14004211
def fisher(params1, params2):
    mu1, sigma1 = params1
    mu2, sigma2 = params2
    fisher_dist = tf.reduce_sum(tf.sqrt(2.) * (tf.sqrt(
        ((mu1 - mu2) ** 2 + 2 * (sigma1 - sigma2) ** 2) * ((mu1 - mu2) ** 2 + 2 * (sigma1 + sigma2) ** 2)) + (
                                          mu1 - mu2) ** 2 + 2 * (sigma1 ** 2 - sigma2 ** 2)) / (
                            4 * sigma1 * sigma2), axis=1)
    return fisher_dist


# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# mu1, sigma1 = original
def kl(params1, params2):
    mu1, sigma1 = params1
    mu2, sigma2 = params2
    kl_div = tf.reduce_sum(tf.log(sigma1 / sigma2) + (sigma2 ** 2 + (mu2 - mu1) ** 2) / (2 * sigma1 ** 2) - 1 / 2,
                           axis=1)
    return kl_div
