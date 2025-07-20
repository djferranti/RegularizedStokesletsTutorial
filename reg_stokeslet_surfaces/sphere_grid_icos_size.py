def sphere_grid_icos_size(factor):
    """
    Sizes an icosahedral grid on a sphere.

    Parameters
    ----------
    factor : int
        Subdivision factor (must be at least 1).

    Returns
    -------
    node_num : int
        Number of nodes in the grid.

    edge_num : int
        Number of edges in the grid.

    triangle_num : int
        Number of triangles in the grid.
    """
    if factor < 1:
        raise ValueError("Subdivision factor must be at least 1")

    node_num = (
        12
        + 10 * 3 * (factor - 1)
        + 10 * (factor - 2) * (factor - 1)
    )

    edge_num = 30 * factor * factor
    triangle_num = 20 * factor * factor

    return node_num, edge_num, triangle_num

