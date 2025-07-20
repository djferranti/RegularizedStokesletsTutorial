# test_drag_sphere_colab.py

import pyvista as pv 
import numpy as np
from numpy.linalg import cond, solve
from math import pi, sqrt
import sys

# Seems that only static plotting is supported by colab at the moment
pv.global_theme.jupyter_backend = 'static'
pv.global_theme.notebook = True
#pv.start_xvfb()

# Add parent directory to path if needed
sys.path.append('..')


from reg_stokeslet_surfaces.triangulatesphereicos import triangulatesphereicos
from reg_stokeslet_surfaces.assemblestokesletmatrix import assemblestokesletmatrix

def test_drag_sphere_colab():
    regularization = 1e-6
    mu = 1
    #factors = [3, 6, 8, 12]
    factors = [4]
    radius = 1

    ell2_errors = np.zeros(len(factors))
    relative_errors = np.zeros(len(factors))
    average_distance = np.zeros(len(factors))

    for i, ith_factor in enumerate(factors):
        TriangleArray, spherePoints, faces = triangulatesphereicos(ith_factor, radius)

        xField = spherePoints
        number_triangle_points = spherePoints.shape[1]

        print("************Triangle Data******************")
        print(f"number of triangles = {len(TriangleArray)}")
        print(f"number of faces = {faces.shape[1]}")
        print(f"number of DOF (3 times number of sphere vertices) = {3 * number_triangle_points}")
        print("*******************************************")
        A = assemblestokesletmatrix(xField, TriangleArray, number_triangle_points, regularization, mu)
#         print(A)

        #print(f"condition number of Stokeslet matrix = {cond(A)}")

        unit_x = np.array([1, 0, 0])
        uField = np.tile(unit_x, number_triangle_points)
        
#         print(uField)

        F = solve(A, uField)
        
        F_col = F.reshape((number_triangle_points,3)).T 
        
        
#         print(F)
        
        faces = faces.astype(int)

        F0 = F_col[:, faces[0, :]]
        F1 = F_col[:, faces[1, :]]
        F2 = F_col[:, faces[2, :]]

        bh = np.array([T['bh'] for T in TriangleArray])
        total_drag = np.sum((F0 + F1 + F2) / 3 * bh / 2, axis=1)

        expected_drag = np.array([6 * pi * mu * radius, 0, 0])
        error = total_drag - expected_drag

        ell2_errors[i] = np.sqrt(np.sum(error ** 2))
        relative_errors[i] = abs(1 - total_drag[0] / expected_drag[0]) * 100
        average_distance[i] = np.sqrt(np.mean(bh))

        print("Drag error data")
        print(f"ell2 error = {ell2_errors[i]}")
        print(f"relative error (%) = {relative_errors[i]}")
        print(f"average distance = {average_distance[i]}")
        
        # display streamlines and sphere 
        
        
        vertices = spherePoints.T 
        faces = faces.T 
        # PyVista expects faces in a flat format: [3, i0, i1, i2, 3, i3, i4, i5, ...]
        faces_flat = np.hstack([np.insert(face, 0, 3) for face in faces]).astype(np.int32)

        
        # Create mesh and plot
        sphere = pv.PolyData(vertices, faces_flat) 
        bounds = sphere.bounds
        
        Ng = 10
        x = np.linspace(bounds[0] - 0.5, bounds[1] + 0.5, Ng)
        y = np.linspace(bounds[2] - 0.5, bounds[3] + 0.5, Ng)
        z = np.linspace(bounds[4] - 0.5, bounds[5] + 0.5, Ng) 
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        Xflatten = X.ravel() 
        Yflatten = Y.ravel()
        Zflatten = Z.ravel() 
        
        original_shape = (Ng, Ng, Ng)
        
        xFieldStream = np.vstack([Xflatten,Yflatten,Zflatten]) 
        
        Astream = assemblestokesletmatrix(xFieldStream, TriangleArray, 
                                          number_triangle_points, regularization, mu)
        
        uStream = Astream @ F
        uStream = uStream.reshape((Ng ** 3, 3 )).T 
        #U,V,W 
        U = uStream[0,:].reshape(original_shape) - unit_x[0]
        V = uStream[1,:].reshape(original_shape) - unit_x[1]
        W = uStream[2,:].reshape(original_shape) - unit_x[2]
        
        # Convert to PyVista structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        vectors = np.stack((U, V, W), axis=-1)
        grid["vectors"] = vectors.reshape(-1, 3)
        speed = np.linalg.norm(grid["vectors"], axis=1)
        grid["speed"] = speed
        
        arrows = grid.glyph(orient='vectors', scale='speed', factor=0.2)
        
        # Seed points: pick some points on a plane upwind of the sphere
        #seeds = pv.Disc(center=(bounds[0] - 0.5, 0, 0), inner=0.0, outer=1.5, normal=(0, 0, 1), r_res=6, c_res=12)
        
        # Define seed grid in z = 0 plane, behind the sphere
        xseed = np.linspace(1.1, 1.6, 10)  # Along flow direction
        yseed = np.linspace(-1.0, 1.0, 20)
        zseed = np.linspace(-0.5,0.5,4)
        #yseed = np.array([0.0])
        #zseed = np.array([0.25])  # Single z-plane
        
        Xseed, Yseed, Zseed = np.meshgrid(xseed, yseed, zseed, indexing='ij')
        # Flatten and stack into (N, 3) array of points
        seed_points = np.vstack((Xseed.ravel(), Yseed.ravel(), Zseed.ravel())).T
        # Convert to pyvista PolyData
        seeds = pv.PolyData(seed_points) 
        
        # Generate streamlines
        streamlines = grid.streamlines_from_source(
            seeds,
            vectors='vectors',
            initial_step_length=0.05,
            terminal_speed=1e-4
            )

        # turn streamlines into tubes for visibility
        stream_tubes = streamlines.tube(radius=0.01)
        
        # Plot the sphere and streamlines
        plotter = pv.Plotter()
        plotter.add_mesh(sphere, color='lightgrey', opacity=1.0, show_edges=True)
        plotter.add_mesh(stream_tubes, scalars="speed", cmap="viridis", 
                         clim=[speed.min(), speed.max()],
                         scalar_bar_args={"title": "Speed", "vertical": True}, 
                         line_width=2)
        plotter.add_mesh(arrows, color='black')  # white arrows on top
        # Add axes
        plotter.show_axes()  # Show XYZ triad
        plotter.view_isometric()  # Or set custom view
        plotter.view_xy()  # Or set custom view
        plotter.show()
            
    return grid, sphere, stream_tubes

if __name__ == "__main__":
    test_drag_sphere_colab()
