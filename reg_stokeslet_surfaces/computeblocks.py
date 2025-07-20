import numpy as np

def computeblocks(geometryData, t001, t101, t011, 
                  t003, t103, t013, t203, t023, t113, t303, t033, t213, t123, regularization):
    def kronx0x0(t, r1, r2, r3):
      #t = t.reshape(-1, 1)
      out = np.kron(t * r1 * r1, np.diag([1, 0, 0])) \
        + np.kron(t * r2 * r2, np.diag([0, 1, 0])) \
        + np.kron(t * r3 * r3, np.diag([0, 0, 1])) \
        + np.kron(t * r1 * r2, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])) \
        + np.kron(t * r1 * r3, np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])) \
        + np.kron(t * r2 * r3, np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))
      return out

    def kronx0vorw(t, r1, r2, r3, v1, v2, v3):
      #t = t.reshape(-1, 1)
      out = np.kron(t * 2 * v1 * r1, np.diag([1, 0, 0])) \
        + np.kron(t * 2 * v2 * r2, np.diag([0, 1, 0])) \
        + np.kron(t * 2 * v3 * r3, np.diag([0, 0, 1])) \
        + np.kron(t * (v1 * r2 + v2 * r1), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])) \
        + np.kron(t * (v1 * r3 + v3 * r1), np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])) \
        + np.kron(t * (v2 * r3 + v3 * r2), np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))
      return out
    
    # Unpackage geometryData (M x 1 vectors)
    x0 = geometryData['x0']
    r1 = x0[0, :]
    r2 = x0[1, :]
    r3 = x0[2, :]
    #r1 = x0[0, :].reshape(-1, 1)
    #r2 = x0[1, :].reshape(-1, 1)
    #r3 = x0[2, :].reshape(-1, 1)

    # 3 x 1 vectors
    vhat = geometryData['vhat']
    v1, v2, v3 = vhat[0], vhat[1], vhat[2]
    what = geometryData['what']
    w1, w2, w3 = what[0], what[1], what[2]

    # The rest are scalars
    ell1 = geometryData['ell1']
    ell2 = geometryData['ell2']

    # The 3x3 block matrices that appear often
    identity = np.eye(3)
    vvt = np.outer(vhat, vhat)
    wwt = np.outer(what, what)
    vwt = np.outer(vhat, what)
    wvt = vwt.T 

    b0 = np.kron(t001 + regularization**2 * t003 - t101 - regularization**2 * t103, identity) \
         + np.kron(ell1**2 * t203 - ell1**2 * t303, vvt) \
         + np.kron(ell2**2 * t023 - ell2**2 * t123, wwt) \
         + np.kron(ell1 * ell2 * t113 - ell1 * ell2 * t213, vwt + wvt) \
         + kronx0vorw(ell1 * t103 - ell1 * t203, r1, r2, r3, v1, v2, v3) \
         + kronx0vorw(ell2 * t013 - ell2 * t113, r1, r2, r3, w1, w2, w3) \
         + kronx0x0(t003 - t103, r1, r2, r3)

    b1 = np.kron(t101 + regularization**2 * t103 - t011 - regularization**2 * t013, identity) \
         + np.kron(ell1**2 * t303 - ell1**2 * t213, vvt) \
         + np.kron(ell2**2 * t123 - ell2**2 * t033, wwt) \
         + np.kron(ell1 * ell2 * t213 - ell1 * ell2 * t123, vwt + wvt) \
         + kronx0vorw(ell1 * t203 - ell1 * t113, r1, r2, r3, v1, v2, v3) \
         + kronx0vorw(ell2 * t113 - ell2 * t023, r1, r2, r3, w1, w2, w3) \
         + kronx0x0(t103 - t013, r1, r2, r3)

    b2 = np.kron(t011 + regularization**2 * t013, identity) \
         + np.kron(ell1**2 * t213, vvt) \
         + np.kron(ell2**2 * t033, wwt) \
         + np.kron(ell1 * ell2 * t123, vwt + wvt) \
         + kronx0vorw(ell1 * t113, r1, r2, r3, v1, v2, v3) \
         + kronx0vorw(ell2 * t023, r1, r2, r3, w1, w2, w3) \
         + kronx0x0(t013, r1, r2, r3)

    return b0, b1, b2