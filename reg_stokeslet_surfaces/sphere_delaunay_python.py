import numpy as np
from scipy.spatial import ConvexHull

def sphere_delaunay_python(xyz):
    """
    Computes a Delaunay triangulation on the unit sphere using convex hull.

    Parameters:
        xyz: 3 x N numpy array of points on the sphere.

    Returns:
        face_num: integer, number of triangle faces.
        face: 3 x face_num numpy array of triangle vertex indices.
    """
    # Transpose to N x 3 for scipy
    hull = ConvexHull(xyz.T)
    face = hull.simplices.T  # shape (3, face_num) 

    # Flip orientation of each triangle
    #face = face[::-1, :]  # same as face(3:-1:1,:) in MATLAB

    # Rotate each column so smallest index is first
    for j in range(face.shape[1]):
        i1, i2, i3 = face[:, j]
        if i2 < i1 and i2 < i3:
            face[:, j] = [i2, i3, i1]
        elif i3 < i1 and i3 < i2:
            face[:, j] = [i3, i1, i2]

    face_num = face.shape[1]
    #print(f"face number = {face_num}")
    return face_num, face
