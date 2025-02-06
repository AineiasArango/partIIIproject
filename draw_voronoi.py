import matplotlib.pyplot as plt

def create_2d_voronoi_example():
    import numpy as np
    from scipy.spatial import Voronoi, voronoi_plot_2d
    import matplotlib.pyplot as plt

    # Create a central point
    center = np.array([[5, 5]])
    
    # Create shells of points with increasing radius and number of points
    all_shells = []
    radii = [2, 4, 6, 8, 11, 14, 17]  # Radii for each shell
    points_per_shell = [5, 12, 23, 30, 29, 36, 43]  # Number of points in each shell
    
    for r, n_points in zip(radii, points_per_shell):
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        shell = np.column_stack([
            5 + r * np.cos(theta),
            5 + r * np.sin(theta)
        ])
        all_shells.append(shell)
    
    # Combine all points
    points = np.vstack([center] + all_shells)

    # Create the Voronoi tessellation
    vor = Voronoi(points)

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot Voronoi diagram with points hidden and thicker lines
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, point_size=0, line_width=4)
    
    # Plot central point as a bullet point
    plt.plot(center[0,0], center[0,1], 'ko', markersize=8)

    # Remove all axes, labels, and borders
    plt.axis('off')
    plt.title('')
    plt.gca().set_position([0, 0, 1, 1])
    
    # Set tighter limits to zoom in
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    return fig

# Create and show the example
"""os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots/")
fig = create_2d_voronoi_example()
plt.savefig("voronoi_example_small_thick.png", bbox_inches='tight')
plt.show()"""

def create_2d_voronoi_example_central_only():
    import numpy as np
    from scipy.spatial import Voronoi
    import matplotlib.pyplot as plt

    # Create a central point
    center = np.array([[5, 5]])
    
    # Create shells of points with increasing radius and number of points
    all_shells = []
    radii = [2, 4, 6, 8, 11, 14, 17]  # Radii for each shell
    points_per_shell = [5, 12, 23, 30, 29, 36, 43]  # Number of points in each shell
    
    for r, n_points in zip(radii, points_per_shell):
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        shell = np.column_stack([
            5 + r * np.cos(theta),
            5 + r * np.sin(theta)
        ])
        all_shells.append(shell)
    
    # Combine all points
    points = np.vstack([center] + all_shells)

    # Create the Voronoi tessellation
    vor = Voronoi(points)

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot only ridges connected to central cell and its immediate neighbours
    central_region = vor.point_region[0]  # Get region index for central point
    neighbour_regions = set()  # Will store regions that touch the central cell
    
    # Find the regions that share a ridge with the central region
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        if p1 == 0 or p2 == 0:  # If ridge connects to central point
            if p1 == 0:
                neighbour_regions.add(vor.point_region[p2])
            else:
                neighbour_regions.add(vor.point_region[p1])
    
    # Plot ridges
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        if -1 not in vor.ridge_vertices[ridge_idx]:  # Skip infinite ridges
            region1, region2 = vor.point_region[p1], vor.point_region[p2]
            # Draw ridge if it's part of central cell or its neighbours
            if (region1 == central_region or region2 == central_region or
                region1 in neighbour_regions or region2 in neighbour_regions):
                vertices = vor.vertices[vor.ridge_vertices[ridge_idx]]
                plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=4)
    
    # Plot central point as a bullet point
    plt.plot(center[0,0], center[0,1], 'ko', markersize=8)

    # Remove all axes, labels, and borders
    plt.axis('off')
    plt.title('')
    plt.gca().set_position([0, 0, 1, 1])
    
    # Set tighter limits to zoom in
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    return fig

os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots/")
fig = create_2d_voronoi_example_central_only()
plt.savefig("voronoi_example_central_only.png", bbox_inches='tight')
plt.show()