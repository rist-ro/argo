import numpy as np
import numexpr as ne

NUMTOL = 1e-15

def KL(p, q):
    p += NUMTOL
    q += NUMTOL
    return np.sum(p * np.log(p / q), axis=-1)

def BC(p, q):
    p += NUMTOL
    q += NUMTOL
    return -np.log(np.sum(np.sqrt(p) * np.sqrt(q), axis=-1))

def h(p, alpha):
    if alpha==1:
        return np.log(p)
    else:
        return (2/(1-alpha)) * p ** ((1 - alpha) / 2)

def h_prime(p, alpha):
    return p ** (- (1 + alpha) / 2)

def fisher_matrix_and_whatnot(V, p, damping=0.):
    # I is the fisher matrix, beta is the (pushforward of a vector in u -> to a vector in R^n_(alpha))
    # i.e. matrix containing as rows the coordinates of the basis for the fisher matrix (beta) in R^n_(alpha)
    p = p.reshape(-1,1)
    E_V = np.sum(p*V, axis=0).reshape(1,-1)
    DV = V-E_V
    I = np.matmul(DV.T, p*DV) + damping * np.eye(DV.shape[1])

    return I, DV

def beta_fisher_basis(DV, p, alpha):
    beta = h_prime(p, alpha).reshape(-1,1) * p * DV
    return beta

def project_vector_on_basis(c_tilde, beta, I_inv, p, alpha):
    beta_a = p.reshape(-1,1)**alpha * beta
    c_proj = np.matmul(I_inv, np.matmul(beta_a.T, c_tilde))
    return c_proj

def project_vectors_on_basis(c_tilde_batch, beta, I_inv, p, alpha):
    beta_a = p.reshape(-1,1)**alpha * beta
    c_proj_batch = np.matmul(I_inv, np.matmul(beta_a.T, np.transpose(c_tilde_batch)))
    return c_proj_batch.T

def project_away_1vec_component(V):
    D = V.shape[0]
    v_shift = np.sum(V, axis=0).reshape(1, -1)/D
    # now V_tilde = V-v_shift is such that V_tilde \cdot 1vec = 0vec
    return V-v_shift

def project_vectors_away_from_normal(c_tilde_batch, p, alpha):
    n_vec = p**((1-alpha)/2)
    n_vec = n_vec.reshape(1,-1)
    np.testing.assert_array_equal(n_vec.shape[1], c_tilde_batch.shape[1])

    ga_nvec = p.reshape(1,-1)**alpha * n_vec.reshape(1,-1)
    n_vec_norm = np.sum(n_vec * ga_nvec)
    np.testing.assert_approx_equal(n_vec_norm, 1)
    c_n_coeffs = np.sum(c_tilde_batch * ga_nvec.reshape(1,-1), axis=1) / n_vec_norm

    c_proj_batch = c_tilde_batch - c_n_coeffs.reshape(-1,1)*n_vec
    return c_proj_batch

def project_on_basis_from_ps(p1, DV, I_inv, p0, alpha):
    if alpha == 1.:
        p_fact = p0 * (np.log(p1) - np.log(p0))
    else:
        # p_fact = (2./(1-alpha)) * (p0**((1.+alpha)/2.) * p1**((1.-alpha)/2.) - p0 )
        p_fact = (2. / (1 - alpha)) * (np.sqrt(p0) * np.sqrt(p1) * (p0/p1)**(alpha/2) - p0)

    c_proj_batch = np.matmul(I_inv, np.matmul(DV.T, np.transpose(p_fact)))
    return c_proj_batch.T

def scalar_prod_logp0pw_beta_basis(pw, p0, DV, alpha):
    """

    Args:
        pw: a batch of probabilities (row:word, column:chi)
        DV: centered statistics (for p0, to be consistent)
        p0: the central probability on which tangent space to project (row vector)
        alpha: the value of alpha

    Returns:
        scalar product between Logmaps of each point in the batch and the basis of the tangent space
        .. math:: \left< \Log^{(\alpha)_{p_0} p_w}, \beta_i^{(\alpha)} \right>_{\mathbb{R}^n_{(\alpha)}}

    """

    p0 = p0.reshape(1, -1)

    if alpha == 1.:
        p_fact = p0 * (np.log(pw+NUMTOL) - np.log(p0+NUMTOL))
    else:
        # p_fact = (2./(1-alpha)) * (p0**((1.+alpha)/2.) * p1**((1.-alpha)/2.) - p0 )
        p_fact = (2. / (1 - alpha)) * p0 * (((pw)/(p0+NUMTOL))**((1-alpha)/2) - 1)


    #p_fact.shape == (BATCH, DICT)
    ldv_alpha = np.matmul(p_fact, DV)

    #ldv_alpha.shape == (BATCH, d), d small linear space (usually d=300)
    return ldv_alpha


def scalar_prod_logp0pw_beta_basis_npf(pw, p0, DV, alpha):
    """
    From normalized p_fact

    Args:
        pw: a batch of probabilities (row:word, column:chi)
        DV: centered statistics (for p0, to be consistent)
        p0: the central probability on which tangent space to project (row vector)
        alpha: the value of alpha

    Returns:
        scalar product between Logmaps of each point in the batch and the basis of the tangent space
        .. math:: \left< \Log^{(\alpha)_{p_0} p_w}, \beta_i^{(\alpha)} \right>_{\mathbb{R}^n_{(\alpha)}}

    """

    p_fact_normalized, l_scale = get_norm_p_fact(p0, pw, alpha)
    ldv_alpha = np.matmul(p_fact_normalized, DV)

    return ldv_alpha, l_scale

def get_norm_p_fact(p0, pw, alpha):
    p0 = p0.reshape(1, -1)

    if alpha == 1.:
        p_fact = p0 * (np.log(pw + NUMTOL) - np.log(p0 + NUMTOL))
        inf_norm = 1
    else:
        ratio = ne.evaluate("pw / (p0 + NUMTOL)") # (pw / (p0 + NUMTOL))
        max_ratio = np.max(ratio, axis=1).reshape(-1, 1)
        alpha_ratio = ne.evaluate("(ratio / max_ratio) ** ((1 - alpha) / 2)")
        alpha_const = ne.evaluate("1. / max_ratio ** ((1 - alpha) / 2)")
        inf_norm = 2/(1-alpha) * max_ratio**((1 - alpha) / 2)
        p_fact = ne.evaluate("p0 * (alpha_ratio - alpha_const)")

    p_fact_norm = np.sqrt(ne.evaluate("p_fact*p_fact").sum(axis=1)).reshape(-1, 1)

    l_norms = inf_norm*p_fact_norm

    return p_fact / p_fact_norm, l_norms

# def get_norm_p_fact(p0, pw, alpha):
#     p0 = p0.reshape(1, -1)
#
#     if alpha == 1.:
#         p_fact = p0 * (np.log(pw + NUMTOL) - np.log(p0 + NUMTOL))
#     else:
#         ratio = (pw / (p0 + NUMTOL))
#         max_ratio = np.max(ratio, axis=1).reshape(-1, 1)
#         alpha_ratio = (ratio / max_ratio) ** ((1 - alpha) / 2)
#         alpha_const = (1. / max_ratio ** ((1 - alpha) / 2))
#         p_fact = (2. / (1 - alpha)) * p0 * (alpha_ratio - alpha_const)
#
#     p_fact_norm = np.sqrt(ne.evaluate("p_fact*p_fact").sum(axis=1)).reshape(-1, 1)
#     # p_fact_norm = np.linalg.norm(p_fact, axis=1).reshape(-1, 1)
#
#     return p_fact / p_fact_norm