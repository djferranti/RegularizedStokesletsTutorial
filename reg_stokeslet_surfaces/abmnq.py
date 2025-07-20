import numpy as np
from scipy.integrate import dblquad, quad

def abmnq(a001, se10p1, se20p1, sd0p1, se10m1, se20m1, sd0m1, geometryData):
    # Unpackage geometryData (M x 1 vectors)
    x0DotV = geometryData['x0DotV']
    x1DotW = geometryData['x1DotW']
    x2DotD = geometryData['x2DotD']
    R0 = geometryData['R0']
    R1 = geometryData['R1']
    R2 = geometryData['R2']
    ell1 = geometryData['ell1']
    ell2 = geometryData['ell2']
    ell3 = geometryData['ell3']

    # Compute the line integral recursions from formula (2.24)
    q = 1  # corresponds to q from formula, and easier on eyes

    se1p1p1 = 1 / (ell1 ** 2) * (-1 / (q - 2) * (R1 - R0) - ell1 * x0DotV * se10p1)
    se1p2p1 = 1 / (ell1 ** 2) * (-1 / (q - 2) * R1 + 1 / (q - 2) * se10m1 - ell1 * x0DotV * se1p1p1)

    se2p1p1 = 1 / (ell2 ** 2) * (-1 / (q - 2) * (R2 - R1) - ell2 * x1DotW * se20p1)
    se2p2p1 = 1 / (ell2 ** 2) * (-1 / (q - 2) * R2 + 1 / (q - 2) * se20m1 - ell2 * x1DotW * se2p1p1)

    sdp1p1 = 1 / (ell3 ** 2) * (-1 / (q - 2) * (R0 - R2) - ell3 * x2DotD * sd0p1)
    sdp2p1 = 1 / (ell3 ** 2) * (-1 / (q - 2) * R0 + 1 / (q - 2) * sd0m1 - ell3 * x2DotD * sdp1p1)

    # Compute the Amnq, Bmnq from formulas (2.22), (2.23)
    a101 = a001 + sdp1p1
    a201 = a001 + 2 * sdp1p1 - sdp2p1
    a011 = se2p1p1 - sd0p1 + sdp1p1
    a021 = se2p2p1 - sd0p1 + 2 * sdp1p1 - sdp2p1

    b101 = -se1p1p1 + sd0p1 - sdp1p1
    b201 = -se1p2p1 + sd0p1 - 2 * sdp1p1 + sdp2p1
    b011 = sd0p1 - sdp1p1
    b021 = sd0p1 - 2 * sdp1p1 + sdp2p1

    return a101, a011, a201, a021, b101, b011, b201, b021

def nIntegrateA001(x0DotV, x0DotW, vDotW, ell1, ell2, R0):
    loopMax = R0.shape[0]
    out = np.zeros_like(R0)

    for i in range(loopMax):
        Rsq = lambda a, b: (x0DotV[i] + a * ell1) ** 2 + (x0DotW[i] + b * ell2) ** 2 + \
                           2 * ell1 * ell2 * a * b * vDotW + R0[i] ** 2 - (x0DotV[i]) ** 2 - (x0DotW[i]) ** 2
        am01 = lambda a, b: -((x0DotV[i] + ell1 * a) * ell1 + ell1 * ell2 * b * vDotW) * Rsq(a, b) ** (-3 / 2)
        bmax = lambda a: a
        jacob = np.sqrt(1 - vDotW ** 2)
        out[i] = (ell1 * ell2 * jacob) * dblquad(am01, 0, 1, 0, bmax)[0]

    return out

def nIntegrateAm01(m, x0DotV, x0DotW, vDotW, ell1, ell2, R0):
    loopMax = R0.shape[0]
    out = np.zeros_like(R0)

    for i in range(loopMax):
        Rsq = lambda a, b: (x0DotV[i] + a * ell1) ** 2 + (x0DotW[i] + b * ell2) ** 2 + \
                           2 * ell1 * ell2 * a * b * vDotW + R0[i] ** 2 - (x0DotV[i]) ** 2 - (x0DotW[i]) ** 2
        am01 = lambda a, b: m * a ** (m - 1) * Rsq(a, b) ** (-1 / 2) - a ** m * \
                            ((x0DotV[i] + ell1 * a) * ell1 + ell1 * ell2 * b * vDotW) * Rsq(a, b) ** (-3 / 2)
        bmax = lambda a: a
        jacob = np.sqrt(1 - vDotW ** 2)
        out[i] = (ell1 * ell2 * jacob) * dblquad(am01, 0, 1, 0, bmax)[0]

    return out

def nIntegrateLineAlphaRm1(x0DotV, ell1, R0):
    loopMax = R0.shape[0]
    out = np.zeros_like(R0)

    for i in range(loopMax):
        Rsq = lambda a: (x0DotV[i] + a * ell1) ** 2 + R0[i] ** 2 - (x0DotV[i]) ** 2
        Rm1 = lambda a: a * Rsq(a) ** (-1/2)
        out[i] = ell1 * quad(Rm1, 0, 1)[0]
    
    return out 

def nIntegrateLineAlpha2Rm1(x0DotV, ell1, R0):
    loopMax = R0.shape[0]
    out = np.zeros_like(R0)

    for i in range(loopMax):
        Rsq = lambda a: (x0DotV[i] + a * ell1) ** 2 + R0[i] ** 2 - (x0DotV[i]) ** 2
        Rm1 = lambda a: a ** 2 * Rsq(a) ** (-1/2)
        out[i] = ell1 * quad(Rm1, 0, 1)[0]
    
    return out 