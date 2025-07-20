import numpy as np

def computet001side(s0p1, x0DotN, sideLength):
    """
    COMPUTET001SIDE computes contribution from side in contour integral part of T001

    Parameters:
        s0p1 (ndarray): M x 1 vector representing the line integral S_{0,1} for each field point.
        x0DotN (ndarray): M x 1 vector representing x0 ⋅ n for every field point, 
                          where n is the unit outward normal to the side.
        sideLength (float): Length of the side.

    Returns:
        t001Side (ndarray): M x 1 vector representing contribution from the side 
                            in the contour integral part of T001.
    """
    s0p1 = np.asarray(s0p1).flatten()
    x0DotN = np.asarray(x0DotN).flatten()

    t001Side = sideLength * (-1) * (x0DotN * s0p1)
    t001Side[np.abs(x0DotN) < np.finfo(float).eps] = 0

    return t001Side
