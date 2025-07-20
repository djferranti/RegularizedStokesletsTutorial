import numpy as np
from .tmp1recursion import tmp1recursion
from .tnp1recursion import tnp1recursion

def tqequals1(se1m1, se2m1, sdm1, t001, geometryData):
    """
    TQEQUALS1 outputs T101, T011 from recursion formula.

    Parameters:
        se1m1 (ndarray): M x 1 vector storing S_{0,1} along e₁.
        se2m1 (ndarray): M x 1 vector storing S_{0,1} along e₂.
        sdm1  (ndarray): M x 1 vector storing S_{0,1} along d.
        t001  (ndarray): M x 1 vector storing T_{0,0,1}.
        geometryData (dict): Dictionary containing geometry data:
            - 'x0DotV', 'x0DotW', 'vDotW', 'ell1', 'ell2', etc.

    Returns:
        t101 (ndarray): M x 1 vector storing T_{1,0,1}.
        t011 (ndarray): M x 1 vector storing T_{0,1,1}.
    """
    se1m1 = np.asarray(se1m1).flatten()
    se2m1 = np.asarray(se2m1).flatten()
    sdm1 = np.asarray(sdm1).flatten()
    t001 = np.asarray(t001).flatten()
    
    M = len(t001)

    # Formulas (2.22) and (2.23)
    a00m1 = se2m1 - sdm1
    b00m1 = -se1m1 + sdm1

    # These are not used in this recursion step
    tm10m2 = np.zeros(M)
    t0m1m2 = np.zeros(M)

    # Recursion indices
    m = 0
    n = 0
    q = 1

    t101 = tmp1recursion(m, n, q, a00m1, b00m1, tm10m2, t0m1m2, t001, geometryData)
    t011 = tnp1recursion(m, n, q, a00m1, b00m1, tm10m2, t0m1m2, t001, geometryData)

    return t101, t011
