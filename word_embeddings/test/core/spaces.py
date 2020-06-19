import scipy
from scipy import optimize
import numpy as np
from numpy import linalg
import operator, os, itertools
from abc import ABC, abstractmethod
import numexpr as ne
ne.set_num_threads(20)
import sys
import gc

def softmax2D_numexpr(xi, norm_counts_cols = False):
    """Compute 2D softmax values, for a list of sets of scores xi. Uses numexpr, supposedly faster than numpy on large arrays."""
    xi_maxes = ne.evaluate('max(xi, axis=1)')[:,None]
    e_xi = ne.evaluate('exp(xi-xi_maxes)')
    del xi_maxes
    gc.collect()

    if norm_counts_cols:
        e_xi = e_xi / ne.evaluate('sum(e_xi, axis=0)').reshape(1, e_xi.shape[1])

    sums = ne.evaluate('sum(e_xi, axis=1)')[:,None]
    # smax = ne.evaluate('e_xi / sums')
    return ne.evaluate('e_xi / sums')

def C_numexpr(u_embeddings, v_embeddings):
    """Compute counts C from U and V"""
    # import pdb; pdb.set_trace()
    # return np.exp(np.matmul(u_embeddings, np.transpose(v_embeddings)))
    xi = np.matmul(u_embeddings, np.transpose(v_embeddings))
    #
    # # xi_maxes = ne.evaluate('max(xi, axis=1)')
    # # e_xi = ne.evaluate('exp(xi-xi_maxes)')
    # # del xi_maxes
    # # gc.collect()
    return ne.evaluate('exp(xi)')


def softmax2D(xi):
    """Compute 2D softmax values, for a list of sets of scores xi."""
    e_xi = np.exp(xi-np.max(xi, axis=1)[:,None])
    return e_xi / np.sum(e_xi, axis=1)

def softmax(xi):
    """Compute softmax values for a set of scores in xi."""
    e_xi = np.exp(xi - np.max(xi))
    return e_xi / e_xi.sum(axis=0)

def calculate_mu_embeddings(u_embeddings, v_embeddings, norm_counts_cols = False):
    """ each row of u_embeddings and of v_embeddings represents an observation """
    mu = softmax2D_numexpr(np.matmul(u_embeddings, np.transpose(v_embeddings)), norm_counts_cols = norm_counts_cols)
    # mu = softmax2D(np.matmul(u_embeddings, np.transpose(v_embeddings)))
    return mu

def calculate_x_embeddings(u_embeddings, v_embeddings):
    mu = calculate_mu_embeddings(u_embeddings, v_embeddings)
    x_embeddings = ne.evaluate('sqrt(mu)')
    # to ensure that they are normalized (due to small numerical errors on the way they are not perfectly normalized)
    norms = np.linalg.norm(x_embeddings, axis=1)
    return x_embeddings/norms.reshape(-1,1)

def send_u_to_x_on_the_sphere(u, v_embeddings):
    xi=np.matmul(v_embeddings, u)
    mu=softmax(xi)
    x=np.sqrt(mu)
    return x

#tolerance used for numerical errors
difftolerance=1e-7

def dist_sphere(x_a, x_b):
    if abs(linalg.norm(x_a) - 1.) > difftolerance or abs(linalg.norm(x_b) - 1.) > difftolerance:
        raise ValueError("norm of the vectors passed is not 1. Received vectors: \n" + str(x_a) + "\n" + str(x_b))

    return np.arccos(np.dot(x_a, x_b))

def dist_sphere_riem_amb(xa, xb, g_amb, check_norms=True):
    # x_a and x_b are a batch of examples
    np.testing.assert_equal(g_amb.shape[0], g_amb.shape[1], "metric of the ambient space must be a square matrix, found shape {:}".format(g_amb.shape))

    Ixb = np.matmul(g_amb, xb.T).T

    if check_norms:
        sqnormb = np.sum(xb * Ixb, axis=1)

        Ixa = np.matmul(g_amb, xa.T).T
        sqnorma = np.sum(xa * Ixa, axis=1)
        if any(abs(np.sqrt(sqnorma) - 1.) > difftolerance) or any(abs(np.sqrt(sqnormb) - 1.) > difftolerance):
            import pdb;pdb.set_trace()
            raise ValueError("norm of the vectors passed is not 1 respect to the metric of the ambient space. Received norms: {:} \n {:}\n".format(sqnorma, sqnormb))

    return np.arccos(np.matmul(xa, Ixb.T))

def distance_on_the_sphere(u_a, u_b, v_embeddings):
    # proj_xa_of_xb_min_xa = projection_on_the_sphere(x_a, x_b-x_a)
    # logmap = (np.arccos(np.dot(x_a,x_b)) / np.linalg.norm(proj_xa_of_xb_min_xa)) * proj_xa_of_xb_min_xa
    x_a=send_u_to_x_on_the_sphere(u_a, v_embeddings)
    x_b=send_u_to_x_on_the_sphere(u_b, v_embeddings)
    return np.arccos(np.dot(x_a,x_b))

def mean_on_sphere(xp):
    dim = xp.shape[1]
    space = HyperSphere(dim)
    return space.mean(xp)

NUMTOL=1e-23

def logmap_on_the_sphere(x_a, x_b):
    #take two points on the sphere and calculate Log_{x_a} x_b
    
    with np.errstate(all='raise'):
        proj_xa_of_xb_min_xa = projection_on_the_sphere(x_a, x_b-x_a)
        if np.linalg.norm(proj_xa_of_xb_min_xa)>NUMTOL:
            logmap = (np.arccos(np.dot(x_a,x_b)) / np.linalg.norm(proj_xa_of_xb_min_xa)) * proj_xa_of_xb_min_xa
        else:
            logmap = proj_xa_of_xb_min_xa
    
    return logmap

def projection_on_the_sphere(x, Avec):
    return Avec-(np.dot(x,Avec))*x

def distance_in_euclidean_space(u1, u2):
    return np.linalg.norm(u2-u1)

def check_in_tangent_space_sphere(x0, v0):
    if np.dot(x0, v0)>difftolerance:
        raise ValueError("The vector v0: ", v0, " does not belong to the tangent space of x0: ", x0,
                        "their scalar product is: ", np.dot(x0, v0))

def parallel_transport_on_the_sphere(B0, x0, v0):
    """ in x(0) I have the vector B(0), I move with a geodesics on the sphere, in the direction v(0) """
    check_in_tangent_space_sphere(x0, v0)
    check_in_tangent_space_sphere(x0, B0)
    
    with np.errstate(all='raise'):
        omega = np.linalg.norm(v0)
        if omega>NUMTOL:
            u0 = v0/omega
            Bt = np.dot(B0,u0)*(u0*np.cos(omega) - x0*np.sin(omega)) + (B0 - np.dot(u0,B0)*u0)
        else:
            Bt = B0
    
    return Bt

def geodesics_on_the_sphere(x0, v0):
    #I am in x(0) I move with a geodesics on the sphere, in the direction v(0)
    check_in_tangent_space_sphere(x0, v0)
    omega = np.linalg.norm(v0)
    u0 = v0/omega
    xt = x0*np.cos(omega) + u0*np.sin(omega)
    return xt


class Space(ABC):

    def __init__(self, dim):
        self._dim = dim
        self._x0 = self._get_zero()
    
    def median(self, points, **fpkwargs):
        """ Calculate the median in the Space (with the fixed point method).
        The median is meant the point minimizing the sum of the distances.
        
        Args:
            points (ndarray) : 2D array, representing a set of points.
                Each row is a point in the Space.

        Returns:
            ndarray : The median point in the Space.

        """
        
        return self._fixed_point(points, self._psif1, **fpkwargs)
    
    
    def mean(self, points, **fpkwargs):
        """ Calculate the mean in the Space (with the fixed point method).
        The mean is meant the point minimizing the sum of the squared distances.
        
        Args:
            points (ndarray) : 2D array, representing a set of points.
                Each row is a point in the Space.

        Returns:
            ndarray : The mean point in the Space.

        """
        return self._fixed_point(points, self._psif2, **fpkwargs)
    
    
    def dist_hist(self, points, xbar, **histkwargs):
        """Calculate the distances in the space and then make a histogram.

        Args:
            points (type): 2D array, representing a set of points.
                Each row is a point in the Space.
            xbar (type): 1D (or 2D) array. `reference` argument of `space.dist`.
            **histkwargs (type): Optional arguments of `np.histogram` in dictionary form.

        Returns:
            sqrt_mean_sq_dist -> the sqrt of the mean of the squared distances
            dmin, dmax, dmean, dstd -> min, max, mean and std of the distances
            histogram of the distances, as returned by `np.histogram`

        """
        dists = self.dist(points, reference=xbar)
        dmean = np.mean(dists)
        diff = dists-dmean
        dvar = np.dot(diff, diff) / len(diff)
        sqrt_mean_sq_dist = np.sqrt(np.sum(dists**2)/len(dists))
        
        dstd = np.sqrt(dvar)
        dmin = np.min(dists)
        dmax = np.max(dists)
        
        bins = histkwargs.get('bins', 'auto')
        histkwargs.update({'bins' : bins})
        hi = np.histogram(dists, **histkwargs)
        
        return sqrt_mean_sq_dist, dmin, dmax, dmean, dstd, hi
    
    def analogy_measure(self, x_a, x_b, x_c, x_d, x_0=None):
        """Evaluate analogy measure of a:b=c:d
        the analogy measure is calculated in x_0, since all the vectors
        must be in the same tangent space to be compared.

        Args:
            x_a : embedding of the word a involved in the word analogy.
            x_b : embedding of the word b involved in the word analogy.
            x_c : embedding of the word c involved in the word analogy.
            x_d : embedding of the word d involved in the word analogy.
            x_0 : point of the space where to evaluate the analogy measure.

        Returns:
            float: word analogy measure.

        """
        
        # if x_0 is not given explicitly, x_0 is set to the zero of the space,
        # often corresponding to uniform probability distribution
        # e.g. in the simplex: mu0=[1/D,1/D, ... , 1/D] -> x0=sqrt(mu0)
        if x_0 is None:
            x_0 = self._x0
        
        ptra0_logab = self.parallel_transport(self.logmap(x_a, x_b), x_a, self.logmap(x_a, x_0))
        ptrc0_logcd = self.parallel_transport(self.logmap(x_c, x_d), x_c, self.logmap(x_c, x_0))
        return np.linalg.norm(ptrc0_logcd - ptra0_logab)
    
    def _fixed_point(self, points, psif, **fpkwargs):
        self._check_points(points)
        x0 = self._x0_fp(points)
        method = fpkwargs.get('method', 'iteration')
        fpkwargs.update({'method' : method})
        xhat = scipy.optimize.fixed_point(psif, x0, args=[points], **fpkwargs)
        return xhat
    
    def _check_points(self, points):
        assert len(points.shape)==2, "`points` must be a 2D ndarray"
    
    def _check_reference(self, reference):
        assert (len(reference.shape)==np.array([1,2])).any(), "`reference` must be either a 1D or a 2D ndarray"
    
    @abstractmethod
    def dist(self, points, reference=None):
        """Calculate distances in the Space.

        Args:
            points (ndarray) : 2D array, representing a set of points.
                Each row is a point in the Space.

            reference (ndarray): 1D (or 2D) array. The reference point(s) to
                calculate the distances from.

        Returns:
            ndarray: array of distances.

        """
        
        self._check_points(points)
        
        if reference is not None:
            self._check_reference(reference)
    
    @abstractmethod
    def logmap(self, x_a, x_b):
        """Logarithmic map.

        Args:
            x_a (type): Origin in the space.
            x_b (type): Destination in the space.

        Returns:
            ndarray: Vector to go from origin to destination.

        """
        pass
    
    @abstractmethod
    def parallel_transport(self, B0, x0, v0):
        """ in x(0) I have the vector B(0), I move with a geodesics in the space, in the direction v(0) """
        pass
    
    @abstractmethod
    def geodesic(self, x0, v0):
        """ I am in x(0) I move with a geodesic on the sphere, in the direction v(0) """
        pass

    @abstractmethod
    def _x0_fp(self, points):
        pass

    @abstractmethod
    def _get_zero(self):
        """the 0 of this space (often corresponding to the uniform distribution)"""
        pass
    
    def _psif2(self, x, points):
        """ fixed point function for the point xhat, minimizing the sum of the squared distances """
        raise RuntimeError("the function `_psif2` has been called before being defined, either redefine mean or redefine this function.")
        
    def _psif1(self, x, points):
        """ fixed point function for the point xhat, minimizing the sum of the distances """
        raise RuntimeError("the function `_psif1` has been called before being defined, either redefine median or redefine this function.")



class EuclideanSpace(Space):
    """ An Euclidean Space (R^n) """
    
    def dist(self, points, reference=None):
        super().dist(points, reference=reference)
        
        if reference is None:
            reference = np.zeros(points.shape[1])
                
        return np.linalg.norm(points-reference, axis=1)
    
    def mean(self, points):
        self._check_points(points)
        return np.mean(points, axis=0)
    
    def _psif1(self, x, points):
        dists = self.dist(points, reference=x)
        return np.sum(points/dists.reshape((-1,1)), axis=0) / np.sum(1/dists)
    
    def _x0_fp(self, points):
        return self.mean(points)

    def _get_zero(self):
        return np.zeros(self._dim)

    def logmap(self, x_a, x_b):
        return x_b - x_a

    def parallel_transport(self, B0, x0, v0):
        return B0
    
    def geodesic(self, x0, v0):
        return x0 + v0


class HyperSphere(Space):
    """ An embedded n-dimensional Hypersphere (embedded in R^(n+1))
    
    for more info on the methods used refer to:
    [1] Huckermann Ziezold (2006). PCA for Riemannian Manifolds with an Application to Triangular Shape Spaces
    
    """

    def dist(self, points, reference=None):
        super().dist(points, reference=reference)
        
        if reference is None:
            nplusone = points.shape[1]
            reference = np.ones(nplusone)/nplusone
        
        if np.any(abs(np.linalg.norm(points, axis=1)-1.)>difftolerance) or \
            np.any(abs(np.linalg.norm(reference, axis=1)-1.)>difftolerance):
            raise ValueError("norm of some of the vectors passed is not 1.")
        
        scalprods = np.sum(points*reference, axis=1)
        if np.any(abs(scalprods)-1.>difftolerance):
            raise ValueError("Some scalar products are more than 1 or less than -1.")
        
        #for numerical errors some of these products might accidentally be slightly more than 1 and slightly less than -1
        scalprods = np.clip(scalprods, -1, 1)
        
        with np.errstate(all='raise'):
            dist=np.arccos(scalprods)
        # except:
        #     import pdb;pdb.set_trace()
        #
        return dist

    def _psif2(self, x, points):
        zita = np.matmul(points, x)
        xi = np.arccos(zita) / np.sqrt(1-zita**2)
        psi = np.sum(points * xi.reshape((-1,1)), axis=0)
        return psi/np.linalg.norm(psi)

    def _psif1(self, x, points):
        zita = np.matmul(points, x)
        xi = 1 / np.sqrt(1-zita**2)
        psi = np.sum(points * xi.reshape((-1,1)), axis=0)
        return psi/np.linalg.norm(psi)

    def _x0_fp(self, points):
        x0 = np.mean(points, axis=0)
        x0 = x0/np.linalg.norm(x0)
        return x0

    def _get_zero(self):
        return np.sqrt(np.ones(self._dim+1)/(self._dim+1))

    def logmap(self, x_a, x_b):
        return logmap_on_the_sphere(x_a, x_b)

    def parallel_transport(self, B0, x0, v0):
        return parallel_transport_on_the_sphere(B0, x0, v0)
    
    def geodesic(self, x0, v0):
        return geodesics_on_the_sphere(x0, v0)
    

class EmbeddingsManager(ABC):
    
    def __init__(self, dictionary, reversed_dictionary, embeddings, space, extra_info=None):
        """
        Args:
            dictionary (dict): the dictionary of the words {word:index}.
            reversed_dictionary (dict): the reversed_dictionary {index:word}.
            embeddings (ndarray): The embeddings, first dimension must match with dictionary size.
            space (spaces.Space): the space in which the embeddings live.
            extra_info (list): a general purpose variable thought as a list of extra info.
                        For example I might wanna remember u_embeddings and v_embeddings
                        separately in case I use embeddings in the sphere

        """
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary
        self.dictionary_size = len(dictionary)
        self.embeddings = embeddings
        self.extra_info = extra_info
        
        if len(self.embeddings)!=self.dictionary_size:
            raise ValueError("the number of embeddings passed %d is different from the dictionary length %d"%(len(self.embeddings), self.dictionary_size))
        self.space = space
    
    def word_index(self, word):
        try:
            return self.dictionary[word]
        except KeyError as kerr:
            print("\nKey Error: {0}".format(kerr))
            print("The word requested is not present in the dictionary.\n")
            sys.exit(-1)
    
    def index_and_measures(self, index_a, index_b, index_d, howmany=10, amonghowmany=None, tolerance=False):
        """given three indexes a,b,d I want to find c such that a:b=c:d.

        Args:
            index_a, index_b, index_d (int): indexes of the words in the analogy.
            howmany (int): how may words to give as an answer (scored according to distance from target).
            amonghowmany (int): how many words in the dictionary to consider for the answer (the embeddings in the dictionary are ordered by frequency).
            tolerance (boolean): if tolerance, remove words in the query from the possible answers.

        Returns:
            list of (index, embedding): list of answers for the word c in the analogy, sorted by relevance.

        """
        embeddings_to_search = self.embeddings[:amonghowmany]
        x_a=self.embeddings[index_a]
        x_b=self.embeddings[index_b]
        x_d=self.embeddings[index_d]
        #if tolerance, remove words in the query from the possible answers
        #TODO check, is this correct, does embedding numbering start from 0
        xc_star = sorted(
                [(i,self.space.analogy_measure(x_a, x_b, x_c, x_d)) \
                    for (i, x_c) in enumerate(embeddings_to_search) \
                        if (tolerance==False) or (i not in (index_a, index_b, index_d))],
                        key = operator.itemgetter(1)
                        )[:howmany]
        return xc_star
    
    def analogy_query_c(self, word_a, word_b, word_d, **kwargs):
        """given three words a,b,d I want to find c such that a:b=c:d."""
        index_a=self.word_index(word_a)
        index_b=self.word_index(word_b)
        index_d=self.word_index(word_d)
        iam=self.index_and_measures(index_a, index_b, index_d, **kwargs)
        return iam
    
    def analogy_query_d(self, word_a, word_b, word_c, **kwargs):
        """given three words a,b,c I want to find d such that a:b=c:d."""
        # Uses the fact that a:b=c:? => b:a=?:c.
        return self.analogy_query_c(word_b, word_a, word_c, **kwargs)
    
    def analogy_query_d_from_indexes(self, ia, ib, ic, **kwargs):
        """given three word indexes ia,ib,ic I want to find d such that a:b=c:d."""
        # Uses the fact that a:b=c:? => b:a=?:c.
        iam = self.index_and_measures(ib, ia, ic, **kwargs)
        return iam
    
    def word_embeddings(self, words):
        """Return a list of embeddings for the specified words"""
        return np.array([self.embeddings[self.word_index(w)] for w in words])
    
    def distances(self, words1, words2):
        """Distances between two lists of words, according to the embeddings of the EmbeddingsManager.

        Args:
            words1 (list of strings):
            words2 (list of strings):

        Returns:
            ndarray : distances between the words

        """
        emb1 = np.array([self.embeddings[self.word_index(w)] for w in words1])
        emb2 = np.array([self.embeddings[self.word_index(w)] for w in words2])
        return self.space.dist(emb1, emb2)
    

class EmbeddingsManagerUV(EmbeddingsManager):

    def __init__(self, dictionary, reversed_dictionary, u_embeddings, v_embeddings):
        """
        Args:
            dictionary (dict): the dictionary of the words {word:index}.
            reversed_dictionary (dict): the reversed_dictionary {index:word}.
            u_embeddings (ndarray): The U embeddings, first dimension must match with dictionary size.
            v_embeddings (ndarray): The V embeddings, first dimension must match with dictionary size.

        """
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary
        self.dictionary_size = len(dictionary)
        self.space = HyperSphere(self.dictionary_size-1)
        self.u_embeddings = u_embeddings
        self.v_embeddings = v_embeddings

        if len(self.u_embeddings)!=self.dictionary_size or len(self.v_embeddings)!=self.dictionary_size:
            raise ValueError("the number of embeddings passed U:%d and V:%d are different from the dictionary length %d"
                            %(len(self.u_embeddings), len(self.v_embeddings), self.dictionary_size))

        self.embeddings = self.u_embeddings

        # self.scalarprods = np.matmul(u_embeddings, np.transpose(v_embeddings))
        
    
# Distance calculators are deprecated use EmbeddingManagers instead
# class DistanceCalculator(ABC):
#
#     def __init__(self, dictionary, reversed_dictionary, u_embeddings, v_embeddings):
#         self.dictionary=dictionary
#         self.reversed_dictionary=reversed_dictionary
#         self.dictionary_size=len(dictionary)
#         self.u_embeddings=u_embeddings
#         self.v_embeddings=v_embeddings
#
#     def word_index(self, word):
#         try:
#             return self.dictionary[word]
#         except KeyError as kerr:
#             print("\nKey Error: {0}".format(kerr))
#             print("The word requested is not present in the dictionary.\n")
#             sys.exit(-1)
#
#     @abstractmethod
#     def index_and_measures(self, index_a, index_b, index_d, howmany=10, amonghowmany=None, tolerance=False):
#         raise ValueError("this method has not been instantiated yet! Doing nothing.")
#
#     def analogy_query_c(self, word_a, word_b, word_d):
#         """given three words a,b,d I want to find c such that a:b=c:d."""
#         index_a=self.word_index(word_a)
#         index_b=self.word_index(word_b)
#         index_d=self.word_index(word_d)
#         iam=self.index_and_measures(index_a,index_b,index_d)
#         return iam
#
#     def analogy_query_d(self, word_a, word_b, word_c):
#         """given three words a,b,c I want to find d such that a:b=c:d."""
#         # Uses the fact that a:b=c:? => b:a=?:c.
#         return self.analogy_query_c(word_b, word_a, word_c)
#
#     def analogy_query_d_from_indexes(self, ia,ib,ic):
#         """given three word indexes ia,ib,ic I want to find d such that a:b=c:d."""
#         # Uses the fact that a:b=c:? => b:a=?:c.
#         iam=self.index_and_measures(ib,ia,ic)
#         return iam
#
# class DistanceCalculatorMeasures(DistanceCalculator):
#
#     def index_and_measures(self, index_a, index_b, index_d, howmany=10, amonghowmany=None, tolerance=False):
#         u_embeddings_to_search = self.u_embeddings[:amonghowmany]
#         u_a=self.u_embeddings[index_a]
#         u_b=self.u_embeddings[index_b]
#         u_d=self.u_embeddings[index_d]
#         #if tolerance, remove words in the query from the possible answers
#         uc_star = sorted([(i,self.analogy_measure(u_a, u_b, uc, u_d, self.v_embeddings)) \
#             for (i,uc) in enumerate(u_embeddings_to_search) if (tolerance==False) or (i not in (index_a,index_b,index_d))],\
#             key=operator.itemgetter(1))[:howmany]
#         return uc_star
#
#     @abstractmethod
#     def analogy_measure(self, u_a, u_b, u_c, u_d, v_embeddings):
#         pass
#
# class DistanceCalculatorMeasuresEuclidean(DistanceCalculatorMeasures):
#
#     def analogy_measure(self, u_a, u_b, u_c, u_d, v_embeddings):
#         ptra0_logab = u_b-u_a
#         ptrc0_logcd = u_d-u_c
#         return np.linalg.norm(ptrc0_logcd - ptra0_logab)
#
# class DistanceCalculatorMeasuresSpherein0(DistanceCalculatorMeasures):
#
#     def __init__(self, dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany=10, amonghowmany=None, tolerance=False):
#         super().__init__(dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany, amonghowmany, tolerance)
#         self.x_0 = np.sqrt(np.ones(self.dictionary_size)/self.dictionary_size)
#
#     def analogy_measure(self, u_a, u_b, u_c, u_d, v_embeddings):
#         x_a=send_u_to_x_on_the_sphere(u_a, v_embeddings)
#         x_b=send_u_to_x_on_the_sphere(u_b, v_embeddings)
#         x_c=send_u_to_x_on_the_sphere(u_c, v_embeddings)
#         x_d=send_u_to_x_on_the_sphere(u_d, v_embeddings)
#         x_0=self.x_0
#         #x_0 is the uniform probability distribution of the simplex, i.e. mu0=[1/D,1/D, ... , 1/D] -> x0=sqrt(mu0)
#         ptra0_logab = parallel_transport_on_the_sphere(logmap_on_the_sphere(x_a,x_b),x_a,logmap_on_the_sphere(x_a,x_0))
#         ptrc0_logcd = parallel_transport_on_the_sphere(logmap_on_the_sphere(x_c,x_d),x_c,logmap_on_the_sphere(x_c,x_0))
#         return np.linalg.norm(ptrc0_logcd - ptra0_logab)
#
# class DistanceCalculatorMeasuresSphereinA(DistanceCalculatorMeasures):
#
#     def analogy_measure(self, u_a, u_b, u_c, u_d, v_embeddings):
#         x_a=send_u_to_x_on_the_sphere(u_a, v_embeddings)
#         x_b=send_u_to_x_on_the_sphere(u_b, v_embeddings)
#         x_c=send_u_to_x_on_the_sphere(u_c, v_embeddings)
#         x_d=send_u_to_x_on_the_sphere(u_d, v_embeddings)
#
#         logab = logmap_on_the_sphere(x_a,x_b)
#         ptrca_logcd = parallel_transport_on_the_sphere(logmap_on_the_sphere(x_c,x_d),x_c,logmap_on_the_sphere(x_c,x_a))
#         return np.linalg.norm(ptrca_logcd - logab)
#
#
# class DistanceCalculatorTarget(DistanceCalculator):
#
#     def __init__(self, dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany=10, amonghowmany=None, tolerance=False):
#         super().__init__(dictionary, reversed_dictionary, u_embeddings, v_embeddings, howmany, amonghowmany, tolerance)
#         self.search_embeddings=self.embeddings_to_search()
#
#     def index_and_measures(self, index_a, index_b, index_d, howmany=10, amonghowmany=None, tolerance=False):
#         target=self.get_target(index_a, index_b, index_d)
#         return self.find_closest(target, self.search_embeddings, index_a, index_b, index_d)
#
#     def find_closest(self, x_o, x_embeddings, index_a, index_b, index_d, howmany, amonghowmany, tolerance):
#         """ x_o is the objective vector, which vector among x_embeddings is the nearest to this one? """
#         #if tolerance, remove words in the query from the possible answers
#         xc_star = sorted([(i,self.dist(x_o,xc)) for (i,xc) in enumerate(x_embeddings[:amonghowmany]) \
#                 if (tolerance==False) or (i not in (index_a,index_b,index_d))], key=operator.itemgetter(1))[:howmany]
#         return xc_star
#
#     @abstractmethod
#     def get_target(self, index_a, index_b, index_d):
#         pass
#
#     @abstractmethod
#     def dist(self, uo, uc):
#         pass
#
#     @abstractmethod
#     def embeddings_to_search(self):
#         pass
#
#
# class DistanceCalculatorTargetEuclidean(DistanceCalculatorTarget):
#
#     def embeddings_to_search(self):
#         return self.u_embeddings
#
#     def get_target(self, index_a, index_b, index_d):
#         return (self.u_embeddings[index_a]-self.u_embeddings[index_b]+self.u_embeddings[index_d])
#
#     def dist(self, uo, uc):
#         return np.linalg.norm(uo - uc)
#
# class DistanceCalculatorTargetSphere(DistanceCalculatorTarget):
#
#     def embeddings_to_search(self):
#         return calculate_x_embeddings(self.u_embeddings, self.v_embeddings)
#
#     def get_target(self, index_a, index_b, index_d):
#         """ follow_logmap_on_the_sphere """
#         x_a=self.search_embeddings[index_a]
#         x_b=self.search_embeddings[index_b]
#         x_d=self.search_embeddings[index_d]
#
#         logba = logmap_on_the_sphere(x_b,x_a)
#         logbd = logmap_on_the_sphere(x_b,x_d)
#         x_target = geodesics_on_the_sphere(x_d, parallel_transport_on_the_sphere(logba, x_b, logbd))
#         return x_target
#
#     def dist(self, xo, xc):
#         return np.arccos(np.dot(xo,xc))
