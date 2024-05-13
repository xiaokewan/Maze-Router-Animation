import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def draw_triangular_hexagonal_grid(m, n):
    G = nx.hexagonal_lattice_graph(m=m, n=n, periodic=False, with_positions=True, create_using=None)
    pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots()
    a = pos[(0,0)][0]
    b = pos[(0,1)][1]
    c = 2*a
    triangle_points = {
        'up_left_down': np.array([[-c, 0], [a, b], [a, -b]]),
        'left': np.array([[-a, b], [-a, -b], [c, 0]]),
        'up_right_down': np.array([[-a, -b], [-a, b], [c, 0]]),
        'right': np.array([[-c, 0], [a, -b], [a, b]]),
    }

    for node, point in pos.items():
        x, y = point
        if node[0] % 2 == 0:
            direction = node[1] % 2 == 0
            triangle = triangle_points['up_left_down'] if direction else triangle_points['up_right_down']
        else:
            triangle = triangle_points['left'] if node[1] % 2 == 0 else triangle_points['right']
        triangle_copy = triangle.copy()
        triangle_copy[:, 0] += x
        triangle_copy[:, 1] += y
        ax.fill(triangle_copy[:, 0], triangle_copy[:, 1], color='white', edgecolor='black')

    nx.draw(G, pos, node_color='white', node_size=0, with_labels=False, ax=ax)
    plt.axis('equal')
    plt.show()

draw_triangular_hexagonal_grid(6, 3)
