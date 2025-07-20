import numpy as np 

from reg_stokeslet_surfaces.computebasecases import computebasecases 
from reg_stokeslet_surfaces.tqequals1 import tqequals1 
from reg_stokeslet_surfaces.tqequals3 import tqequals3
from reg_stokeslet_surfaces.computeblocks import computeblocks


def assemblestokesletmatrix(xField, TriangleArray, numberTrianglePoints, regularization, mu):
    """
    ASSEMBLESTOKESLETMATRIX assembles the regularized Stokeslet surface matrix. 
    Parameters:
        xField: 3 x M array of field points 
        TriangleArray: list of Q dictionaries where Q is the number of triangular faces. 
        numberTrianglePoints: number of unique points that make up triangulation
        regularization: blob parameter
        mu: viscosity parameter
    Output:
        A: 3M x 3V regularized Stokeslet surface matrix which is related to the
        velocity U at the field points xf by U = A*F where F are the forces at 
        the V distinct vertices of the Q triangular faces.
    """
    
    def addBlockColumn(stokesletMatrix, block, index, bh):
        stokesletMatrix[:, 3 * index: 3 * index + 3] += \
            block * (bh / (8 * np.pi * mu)) 
        return stokesletMatrix 
    
    numberFaces = len(TriangleArray)
    numberFieldPoints = xField.shape[1]

    stokesletMatrix = np.zeros((3 * numberFieldPoints, 3 * numberTrianglePoints))
    
    #print(f"shape of stokeslet matrix =  {stokesletMatrix.shape}")

    for q in range(numberFaces):
        # triangle q
        Triangle = TriangleArray[q]
        bh = Triangle['bh']
        
        #print(f"bh = {Triangle['bh']}")

        # compute the base cases
        t003, t001, se1m1, se2m1, sdm1, se1p1, se2p1, sdp1, geometryData = \
            computebasecases(xField, Triangle, regularization)  
        

        # compute the other T_{mnq} recursively
        t101, t011 = tqequals1(se1m1, se2m1, sdm1, t001, geometryData)
        t103, t013, t203, t023, t113, t303, t033, t213, t123 = \
            tqequals3(se1p1, se2p1, sdp1, se1m1, se2m1, sdm1, t001, t003, 
                       t101, t011, geometryData)
        
        #print(t213)

        # compute the block columns for the stokeslet matrix
        b0, b1, b2 = computeblocks(geometryData, t001, t101, t011, 
                                    t003, t103, t013, t203, t023, 
                                    t113, t303, t033, t213, t123, 
                                    regularization)

        # put the blocks into relevant block columns corresponding to indices
        indices = Triangle['indices']
        index0, index1, index2 = indices

        stokesletMatrix = addBlockColumn(stokesletMatrix, b0.T, index0, bh)
        stokesletMatrix = addBlockColumn(stokesletMatrix, b1.T, index1, bh)
        stokesletMatrix = addBlockColumn(stokesletMatrix, b2.T, index2, bh)

    return stokesletMatrix