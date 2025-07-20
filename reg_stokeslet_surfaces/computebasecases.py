import numpy as np

from reg_stokeslet_surfaces.computet003side import computet003side 
from reg_stokeslet_surfaces.computet001side import computet001side 
from reg_stokeslet_surfaces.computes0p1 import computes0p1 
from reg_stokeslet_surfaces.computes0m1 import computes0m1 

import numpy as np

def computebasecases(xField, Triangle, regularization):
    """
    Compute base cases for regularized Stokeslet surface integrals on a triangle.

    Parameters:
        xField : np.ndarray
            3 x M array of field points.
        Triangle : object
            Contains .vertices (3x3), .normalstosides (3x3), .lengths (3,), .directions (3x3), .bh (scalar).
        regularization : float
            Regularization parameter ε.

    Returns:
        t003, t001, se1m1, se2m1, sdm1, se1p1, se2p1, sdp1 : (M,) arrays
        geometryData : dict
    """

    M = xField.shape[1]
    t003 = np.zeros(M)
    t001 = np.zeros(M)
    se1m1 = np.zeros(M)
    se2m1 = np.zeros(M)
    sdm1 = np.zeros(M)
    se1p1 = np.zeros(M)
    se2p1 = np.zeros(M)
    sdp1 = np.zeros(M)

    bh = Triangle['bh']

    R0 = np.zeros(M)
    R1 = np.zeros(M)
    R2 = np.zeros(M)
    x0DotV = np.zeros(M)
    x1DotW = np.zeros(M)
    x2DotD = np.zeros(M)

    ell1 = 0
    ell2 = 0
    ell3 = 0

    for i in range(3):
        ya = Triangle['vertices'][:, i].reshape(3, 1)
        x0i = xField - ya
        nhatSidei = Triangle['normalstosides'][:, i].reshape(3, 1)
        sideiLength = Triangle['lengths'][i]
        vhati = Triangle['directions'][:, i].reshape(3, 1)

        nhatiRep = nhatSidei @ np.ones((1, M))
        vhatiRep = vhati @ np.ones((1, M))

        x0DotNi = np.sum(x0i * nhatiRep, axis=0)
        x0DotVi = np.sum(x0i * vhatiRep, axis=0)

        if i == 0:
            r2Proj = np.sum(x0i * x0i, axis=0) - x0DotNi**2 - x0DotVi**2
            r2Proj[r2Proj < np.finfo(float).eps] = 0
            gamma = np.sqrt(r2Proj + regularization**2)

            what = Triangle['directions'][:, i + 1].reshape(3, 1)
            whatRep = what @ np.ones((1, M))
            x0DotW = np.sum(x0i * whatRep, axis=0)

            vhat = vhati
            vDotW = float(np.dot(vhat[:, 0], what[:, 0]))
            x0 = x0i

        t003 += computet003side(x0DotVi, x0DotNi, gamma, sideiLength)
        s0p1 = computes0p1(x0DotVi, x0DotNi, gamma, sideiLength)
        t001 += computet001side(s0p1, x0DotNi, sideiLength)
        s0m1 = computes0m1(x0DotVi, x0DotNi, gamma, sideiLength)

        if i == 0:
            se1p1 += s0p1
            se1m1 += s0m1
            R0 += np.sqrt(np.sum(x0i * x0i, axis=0) + regularization**2)
            x0DotV += x0DotVi
            ell1 += sideiLength
        elif i == 1:
            se2p1 += s0p1
            se2m1 += s0m1
            R1 += np.sqrt(np.sum(x0i * x0i, axis=0) + regularization**2)
            x1DotW += x0DotVi
            ell2 += sideiLength
        elif i == 2:
            sdp1 += s0p1
            sdm1 += s0m1
            R2 += np.sqrt(np.sum(x0i * x0i, axis=0) + regularization**2)
            x2DotD += x0DotVi
            ell3 += sideiLength

    t003 = t003 / gamma
    t001 = t001 - gamma**2 * t003

    t003 = t003 / bh
    t001 = t001 / bh 

    geometryData = {
        'x0': x0,
        'vhat': vhat,
        'what': what,
        'x0DotV': x0DotV,
        'x0DotW': x0DotW,
        'x1DotW': x1DotW,
        'x2DotD': x2DotD,
        'R0': R0,
        'R1': R1,
        'R2': R2,
        'vDotW': vDotW,
        'ell1': ell1,
        'ell2': ell2,
        'ell3': ell3,
        'bh': bh,
    }

    return t003, t001, se1m1, se2m1, sdm1, se1p1, se2p1, sdp1, geometryData
