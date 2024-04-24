import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import heapq
from matplotlib import pyplot as plt
import os
import networkx as nx


dim_x = 6
dim_y = 6
G = nx.hexagonal_lattice_graph(dim_x, dim_y, periodic=False, with_positions=True)
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos=pos, with_labels=True)
plt.show()