import numpy as np
import warnings

def computet003side(x0DotV, x0DotN, gamma, sideLength):
    # COMPUTET003SIDE evaluates contribution from side in contour integral equivalent to T003
    # Parameters:
    #   x0DotV: M x 1 vector representing x0 dot v for every field point, where 
    #   v is direction from endpoint to start point of side
    #   x0DotN: M x 1 vector representing x0 dot n for every field point, where
    #   n is unit outward normal to 
    #   gamma: M x 1 vector representing regularized distance to plane of
    #   triangle for every field point
    #   sideLength: length of side (scalar)
    #
    # Output:
    #   t003Side: M x 1 vector representing contribution from side in the 
    #   contour integral equivalent to T003

    # from analytic integration formula - for details, see paper
    p = x0DotV / sideLength
    q = np.sqrt((x0DotN / sideLength) ** 2 + (gamma ** 2) / (sideLength ** 2))
    x0DotNneq0 = np.abs(x0DotN) > np.finfo(float).eps
    isInPInterval = (-1 < p) & (p < 0)
    condSet1 = x0DotNneq0 & isInPInterval
    condSet2 = x0DotNneq0 & ~isInPInterval

    # this is different from paper - the paper should have contained a
    # multiplicative factor of L/H which turns it into this.
    integralCoefficient = (-1) * x0DotN / sideLength 

    # the following are used several times in the helper functions below
    r1, r2 = r1r2(p, q)
    ellq = sideLength * q
    onePlus = 1 + gamma / ellq
    oneMinus = 1 - gamma / ellq 
    
    # to avoid getting runtime warnings about dividing by zero, do it like this 
    # these issues get taken care of in the conditional pieces below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sqrtOnePlusMinus = np.sqrt(onePlus / oneMinus )
        sqrtOneMinusPlus = np.sqrt(oneMinus / onePlus )
    
    

    # 1st/2nd formula only modifies t003 when condSet1/2 true, x0DotN neq 0
    t003Side = condSet1 * integralCoefficient * formula230(q, onePlus, sqrtOnePlusMinus, sqrtOneMinusPlus, r1, r2) + \
                condSet2 * integralCoefficient * formula229(q, onePlus, sqrtOnePlusMinus, sqrtOneMinusPlus, r1, r2, p)
    

    t003Side[np.abs(x0DotN) < np.finfo(float).eps] = 0

    # another special case is when 1 - gamma./ellq = 0 
    # in the limit as this goes to zero, the resulting expression is 0
    t003Side[np.abs(oneMinus) < np.finfo(float).eps] = 0
    
    #print(t003Side[np.isnan(t003Side)])

    return t003Side

def r1r2(p, q):
    phi1 = np.arccos(1 / np.sqrt(p ** 2 / q ** 2 + 1)) / 2
    phi2 = np.arccos(1 / np.sqrt((1 + p) ** 2 / q ** 2 + 1)) / 2
    r1 = np.tan(phi1)
    r2 = np.tan(phi2)
    return r1, r2

def formula230(q, onePlus, sqrtOnePlusMinus, sqrtOneMinusPlus, r1, r2):
    # integral for p between -1 and 0 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return 2 / (q * onePlus) * sqrtOnePlusMinus * (np.arctan(r1 * sqrtOneMinusPlus) + np.arctan(r2 * sqrtOneMinusPlus))

def formula229(q, onePlus, sqrtOnePlusMinus, sqrtOneMinusPlus, r1, r2, p):
    # integral for p <= -1 or >= 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sgnfac = np.sign(1 + p)
        sgnfac[np.abs(sgnfac) < np.finfo(float).eps] = -1
        return sgnfac * 2 / (q * onePlus) * sqrtOnePlusMinus * (np.arctan(r2 * sqrtOneMinusPlus) - np.arctan(r1 * sqrtOneMinusPlus))