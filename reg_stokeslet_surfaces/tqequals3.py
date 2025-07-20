import numpy as np
from .tmp1recursion import tmp1recursion
from .tnp1recursion import tnp1recursion
from .abmnq import abmnq

def tqequals3(se10p1, se20p1, sd0p1,
              se10m1, se20m1, sd0m1,
              t001, t003, t101, t011,
              geometryData):
    """
    Evaluates T103, T013, T203, T023, T113, T303, T033, T213, T123.

    Parameters:
        se10p1, se20p1, sd0p1: M x 1 arrays
        se10m1, se20m1, sd0m1: M x 1 arrays
        t001, t003, t101, t011: M x 1 arrays
        geometryData: dictionary with geometry-related data

    Returns:
        Tuple of arrays: t103, t013, t203, t023, t113, t303, t033, t213, t123
    """
    M = t001.shape[0]

    # Formulas (2.22) and (2.23)
    a001 = se20p1 - sd0p1
    b001 = -se10p1 + sd0p1

    a101, a011, a201, a021, b101, b011, b201, b021 = abmnq(
        a001, se10p1, se20p1, sd0p1,
        se10m1, se20m1, sd0m1, geometryData
    )

    # Initialize dummy values for t_{m-1,q-2} or t_{n-1,q-2}
    tmm1qm2Dummy = np.zeros(M)
    tnm1qm2Dummy = np.zeros(M)

    # Recursion step m+n=1
    m, n, q = 0, 0, 3
    t103 = tmp1recursion(m, n, q, a001, b001, tmm1qm2Dummy, tnm1qm2Dummy, t003, geometryData)
    t013 = tnp1recursion(m, n, q, a001, b001, tmm1qm2Dummy, tnm1qm2Dummy, t003, geometryData)

    # Recursion step m+n=2
    m, n = 1, 0
    t203 = tmp1recursion(m, n, q, a101, b101, t001, tnm1qm2Dummy, t103, geometryData)
    t113 = tnp1recursion(m, n, q, a101, b101, t001, tnm1qm2Dummy, t103, geometryData)

    m, n = 0, 1
    t023 = tnp1recursion(m, n, q, a011, b011, tmm1qm2Dummy, t001, t013, geometryData)

    # Recursion step m+n=3
    m, n = 2, 0
    t303 = tmp1recursion(m, n, q, a201, b201, t101, tnm1qm2Dummy, t203, geometryData)
    t213 = tnp1recursion(m, n, q, a201, b201, t101, tnm1qm2Dummy, t203, geometryData)

    m, n = 0, 2
    t033 = tnp1recursion(m, n, q, a021, b021, tmm1qm2Dummy, t011, t023, geometryData)
    t123 = tmp1recursion(m, n, q, a021, b021, tmm1qm2Dummy, t011, t023, geometryData)

    return t103, t013, t203, t023, t113, t303, t033, t213, t123
