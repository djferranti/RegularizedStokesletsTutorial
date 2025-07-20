import numpy as np

def tmp1recursion(m, n, q, aqm2, bqm2, tmm1qm2, tnm1qm2, tmnq, geometryData):
    """
    TMP1RECURSION evaluates recursion formula (2.17) from paper
    
    Parameters:
       m,n,q: recursion indices
       aqm2, bqm2: M x 1 vectors containing the data for the contour integrals
       A_{m,n,q-2}, B_{m,n,q-2}
       tmm1qm2, tnm1qm2, tmnq: M x 1 vectors containing the data for the 
       T_{m-1,n,q-2}, T_{m,n-1,q-2}, and T_{m,n,q} integrals respectively.
       geometryData: struct containing data for x0DotV, x0DotW, vDotW, L1, L2
    
    Output:
      tmp1: M x 1 vector containing the data for the T_{m+1,n,q} integral
    """
    
    # Unpackage geometryData (M x 1 vectors)
    x0DotV = geometryData['x0DotV']
    x0DotW = geometryData['x0DotW']
    # The rest are scalars
    vDotW = geometryData['vDotW']
    ell1 = geometryData['ell1']
    ell2 = geometryData['ell2']

    # Formula (2.17)
    tmp1 = 1 / (vDotW**2 - 1) * (aqm2 / (ell1**2 * (q - 2)) - \
                                 vDotW * bqm2 / (ell1 * ell2 * (q - 2)) - \
                                 m / (ell1**2 * (q - 2)) * tmm1qm2 + \
                                 vDotW * n / (ell1 * ell2 * (q - 2)) * tnm1qm2 + \
                                 (x0DotV - vDotW * x0DotW) / ell1 * tmnq)
    return tmp1
