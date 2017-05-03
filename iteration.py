__doc_ = 'iteration part'_

import numpy
import scipy, scipy.linalg

import arr

def generate_basis(n, p):
    # generate basis, n*n matrices
    return [numpy.diag(numpy.ones(n - i), i) + numpy.diag(numpy.ones(n - i), -i) for i in xrange(p)]

def information_distance(Cov1, Cov2):
    # compute information distance between two covariance matrix
    # L_1^(-1) * Cov2 * L_1^(-*)
    L = numpy.linalg.cholesky(Cov1)
    C = scipy.linalg.solve_triangular(L.T, numpy.eye(L.shape[0]), lower = False)
    C = scipy.linalg.solve_triangular(L, arr.mm(Cov2, C), lower = True)
    eigs = numpy.linalg.eigh(C)[0]
    return numpy.sum(numpy.log(eigs) ** 2)

def reduction(Cov, n):
    return None
