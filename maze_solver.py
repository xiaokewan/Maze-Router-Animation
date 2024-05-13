from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from IPython.display import Image, display
import os
import networkx as nx
from algorithms import *


# grid graph
def create_grid_graph_with_obstacles(dim_x, dim_y, obstacles, diagonal=True):
    G = nx.grid_2d_graph(dim_x, dim_y)
    for obstacle in obstacles:
        G.remove_node(obstacle) if obstacle in G else None
    if diagonal:
        for node in G.nodes:
            x, y = node
            neighbors = [(x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]
            valid_neighbors = [(nx, ny) for nx, ny in neighbors if (nx, ny) in G.nodes]
            G.add_edges_from([(node, neighbor) for neighbor in valid_neighbors])
    weights = {edge: 1 for edge in G.edges()}
    nx.set_edge_attributes(G, weights, 'length')
    return G


def create_hexagonal_graph_with_obstacles(dim_x, dim_y ):
    G = nx.hexagonal_lattice_graph(dim_x,dim_y, periodic=True, with_positions=True)


def animate_maze_router(start, end, grid_size, obstacles=[], router='dijkstra',
                        attr={'direction_factor': 1, 'expect_pathlength': 32, 'diagonal_grid': True}):
    def constraint_update(frame):
        ax.clear()
        plt.axis('off')
        ax.grid(True)
        ax.set_xlim(-0.5, dim_x - 0.5)
        ax.set_ylim(-0.5, dim_y - 0.5)
        # border of the figure
        ax.plot([0, dim_y], [0, 0], color="k")
        ax.plot([0, 0], [0, dim_x], color="k")
        ax.plot([dim_y, dim_y], [0, dim_x], color="k")
        ax.plot([0, dim_y], [dim_x, dim_x], color="k")
        if frame < len(visited_nodes):
            current_visited = visited_nodes[frame]
        else:
            current_visited = visited_nodes[-1]

        # max_distance for nomalize the color
        max_distance = max(visited.values(), default=1)
        for x in range(dim_x):
            for y in range(dim_y):
                if (x, y) in obstacles:
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="black"))
                elif (x, y) in current_visited:
                    distance = current_visited[(x, y)]
                    normalized_distance = distance / max_distance
                    color = plt.cm.Blues(normalized_distance + 0.1)
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color=color))
                    ax.text(y, x, str(distance), ha='center', va='center', color='black')

        # begin and end
        ax.add_patch(Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, color="lime"))
        ax.text(start[1], start[0], 'SOURCE', ha='center', va='center', color='black')
        ax.add_patch(Rectangle((end[1] - 0.5, end[0] - 0.5), 1, 1, color="red"))
        ax.text(end[1], end[0], 'SINK', ha='center', va='center', color='black')

        # extra frames for the path (if exists)
        if path and len(visited_nodes) - 1 <= frame < len(visited_nodes) - 1 + extra_frames:
            for pos in path:
                ax.add_patch(Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="yellow"))
                ax.text(pos[1], pos[0], str(current_visited.get(pos, '')), ha='center', va='center', color='black')

    def update(frame):

        # max_frame = max(visited.values(), default=0)
        if isinstance(visited, dict):
            max_frame = max(visited.values(), default=0)
            frame_limit = visited.get(end, max_frame)
        else:
            max_frame = len(visited)
            frame_limit = max_frame
        # frame_limit = visited.get(end, max_frame)
        frame = min(frame, frame_limit)

        ax.clear()
        plt.axis('off')
        # ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.1)
        ax.set_xlim(-0.5, dim_x - 0.5)
        ax.set_ylim(-0.5, dim_y - 0.5)
        ax.set_xticks([i + 0.5 for i in range(dim_x)])
        ax.set_yticks([i + 0.5 for i in range(dim_y)])
        ax.set_xticklabels([str(i + 1) for i in range(dim_x)])
        ax.set_yticklabels([str(i + 1) for i in range(dim_y)])
        ax.tick_params(axis='both', which='both', length=0)
        max_distance = max(visited.values(), default=1)
        for x in range(dim_x):
            for y in range(dim_y):
                if (x, y) in obstacles:
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="#293241"))  # black
                elif visited.get((x, y), float('inf')) <= frame:
                    distance = visited.get((x, y), 0)
                    normalized_distance = distance / max_distance
                    color = plt.cm.Blues(normalized_distance + 0.1)
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color=color))
                    ax.text(y, x, str(distance), ha='center', va='center', color='black')

        #  when wave front reach, begin drawing and path exists
        if frame >= frame_limit and path:
            for pos in path:
                if visited.get(pos, float('inf')) <= frame:
                    ax.add_patch(Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="#ee6c4d"))
                    ax.text(pos[1], pos[0], str(visited.get(pos, '')), ha='center', va='center', color='black')

        # # start and end label
        # # too squeeze
        ax.add_patch(Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, color="#fb8500"))
        # ax.text(start[1] - 1, start[0] - 1, r'$\mathit{S}_1$', ha='center', va='center', color='black', fontsize=16,
        #         fontweight='bold')
        ax.add_patch(Rectangle((end[1] - 0.5, end[0] - 0.5), 1, 1, color="#fb8500"))
        # ax.text(end[1] + 1, end[0] + 1, r'$\mathit{D}_1$', ha='center', va='center', color='black', fontsize=16,
        #         fontweight='bold')

        # # add color bar
        # cax = fig.add_axes([0.92, 0.125, 0.03, 0.75])
        # norm = Normalize(vmin=0, vmax=max_distance)
        # cb = ColorbarBase(cax, cmap=plt.cm.Blues, norm=norm, orientation='vertical')
        # cb.set_label('Path Length')
        # cb.set_ticks([0, max_distance])
        # cb.set_ticklabels(['0', str(max_distance)])

    def save_last_frame_as_pdf(ax, start, end, obstacles, visited, path, grid_size, file_name):
        ax.clear()
        plt.axis('off')
        dim_x, dim_y = grid_size
        ax.set_xlim(-0.5, dim_x - 0.5)
        ax.set_ylim(-0.5, dim_y - 0.5)
        ax.set_xticks([i + 0.5 for i in range(dim_x)])
        ax.set_yticks([i + 0.5 for i in range(dim_y)])
        ax.set_xticklabels([str(i + 1) for i in range(dim_x)])
        ax.set_yticklabels([str(i + 1) for i in range(dim_y)])
        ax.tick_params(axis='both', which='both', length=0)
        max_distance = max(visited.values(), default=1)
        for x in range(dim_x):
            for y in range(dim_y):
                if (x, y) in obstacles:
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="#293241"))  # black
                elif visited.get((x, y), float('inf')) < float('inf'):
                    distance = visited.get((x, y), 0)
                    normalized_distance = distance / max_distance
                    color = plt.cm.Blues(normalized_distance + 0.1)
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color=color))
                    ax.text(y, x, str(distance), ha='center', va='center', color='black')

        ax.add_patch(Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, color="#fb8500"))
        ax.add_patch(Rectangle((end[1] - 0.5, end[0] - 0.5), 1, 1, color="#fb8500"))
        cax = fig.add_axes([0.92, 0.125, 0.03, 0.75])
        norm = Normalize(vmin=0, vmax=max_distance)
        cb = ColorbarBase(cax, cmap=plt.cm.Blues, norm=norm, orientation='vertical')
        cb.set_label('Path Length')
        cb.set_ticks([0, max_distance])
        cb.set_ticklabels(['0', str(max_distance)])
        if path:
            for pos in path:
                x, y = pos
                if pos != start and pos != end:
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="#ee6c4d"))
                    ax.text(y, x, str(visited.get((x, y), '')), ha='center', va='center', color='black')

        plt.savefig(file_name, format='pdf')
    # def a_star_update(frame):
    #     ax.clear()
    #     plt.axis('off')
    #     ax.set_xlim(-0.5, dim_x - 0.5)
    #     ax.set_ylim(-0.5, dim_y - 0.5)
    #
    #     if frame < len(queue_nodes):
    #         current_visited = queue_nodes[frame]
    #     else:
    #         current_visited = queue_nodes[-1]
    #
    #     # max_distance for normalize the color
    #     max_distance = max(visited.values(), default=1)
    #     for x in range(dim_x):
    #         for y in range(dim_y):
    #             if (x, y) in obstacles:
    #                 ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="black"))
    #             elif (x, y) in current_visited:
    #                 distance = current_visited[(x, y)]
    #                 normalized_distance = distance / max_distance
    #                 color = plt.cm.Blues(normalized_distance + 0.1)
    #                 ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color=color))
    #                 ax.text(y, x, str(distance), ha='center', va='center', color='black')
    #
    #     # Draw nodes in the current iteration's queue
    #     if frame < len(queue_nodes):
    #         current_queue_nodes = queue_nodes[frame]
    #         for node, path1, cost in current_queue_nodes:
    #             ax.add_patch(Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, color="orange"))
    #             ax.text(node[1], node[0], f'{cost}', ha='center', va='center', color='black')
    #
    #     # Draw start and end nodes
    #     ax.add_patch(Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, color="lime"))
    #     ax.text(start[1], start[0], 'SOURCE', ha='center', va='center', color='black')
    #     ax.add_patch(Rectangle((end[1] - 0.5, end[0] - 0.5), 1, 1, color="red"))
    #     ax.text(end[1], end[0], 'SINK', ha='center', va='center', color='black')
    #
    #     # Draw extra frames for the path (if exists)
    #     if path and frame == len(queue_nodes) - 1 + extra_frames:
    #         if (x, y) in current_visited:
    #             matching_tuple = next((item for item in current_visited if item[0] == (x, y)), None)
    #             if matching_tuple:
    #
    #                 distance = matching_tuple[2]
    #                 normalized_distance = distance / max_distance
    #                 color = plt.cm.Blues(normalized_distance + 0.1)
    #                 ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color=color))
    #                 ax.text(y, x, str(distance), ha='center', va='center', color='black')

    dim_x, dim_y = grid_size
    diagonal_grid = attr['diagonal_grid']
    G = create_grid_graph_with_obstacles(dim_x, dim_y, obstacles, diagonal=diagonal_grid)
    # use the nx embedded router, method only can be dijkstra/ bellman-ford
    if router == 'dijkstra' or router == 'bellman-ford' or router == 'bfs':
        path = nx.shortest_path(G, start, end, method='dijkstra')  # 注意这里可能需要根据router调整方法
        visited = bfs_with_distances(G, start)
    elif router == 'a_star':
        df = attr["direction_factor"]
        path, visited, visited_nodes = a_star_viz(G, start, end, direction_factor=df)
    elif router == 'constrained_a_star':
        ep = attr["expect_pathlength"]
        df = attr["direction_factor"]
        path, visited, visited_nodes = constrained_a_star_viz(G, start, end, ep, direction_factor=df)
    else:
        raise Exception("We don't support this search method, you need to define it manually.")

    if path:
        print("found path, start making animation")
    parents = {start: None}
    for node in visited:
        for neighbor in G.neighbors(node):
            if neighbor not in parents:
                parents[neighbor] = node

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('off')
    extra_frames = 25  # 5 secs
    if router == "constrained_a_star":
        ani = FuncAnimation(fig, update, frames=len(visited_nodes) + 1 + extra_frames, interval=300, repeat=True)
        dpi = 60
    elif router == "a_star":
        ani = FuncAnimation(fig, update, frames=len(visited_nodes) + 1 + extra_frames, interval=300, repeat=True)
        dpi = 60
    else:
        ani = FuncAnimation(fig, update, frames=max(visited.values()) + 1 + extra_frames, interval=300, repeat=True)
        dpi = 140
    # file_name = './{}_grid_animation_with_path_and_weights.gif'.format(router)
    file_name = './{}_grid_animation_withs{}.gif'.format(router, "_".join(
        f"{key}={value}" for key, value in attr.items()))
    ani.save(file_name, writer='pillow', dpi=dpi)
    display(Image(filename=file_name))
    file_path = os.path.join(os.getcwd(), file_name)
    os.startfile(file_path)
    print("! Success simulation, animation saved in address")
    save_last_frame_as_pdf(ax, start, end, obstacles, visited, None, grid_size, file_name+'.pdf')
    # save_last_frame_as_pdf(ax, start, end, obstacles, visited, path, grid_size, file_name + '_with_path.pdf')

if __name__ == "__main__":
    start, end = (0, 0), (7, 7)
    # obstacles = [(4, 4), (4, 5), (4, 6), (5, 4), (6, 4), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15),
    #              (11, 16), (11, 17), (11, 18), (10, 0), (10, 1), (10, 2), (10, 3), (9, 3), (8, 3), (7, 3)]
    # obstacles = [(4, 4), (4, 5), (4, 6), (7, 4), (6, 1),(6, 2), (6, 3)]
    obstacles = []
    grid_size = [8, 8]
    # router = 'a_star'
    router = 'constrained_a_star'
    attr = {  # for a_star
        'direction_factor': 1.01,
        'expect_pathlength': 20,
        'diagonal_grid': False
    }  # for constrained a_star

    animate_maze_router(start, end, grid_size, obstacles, router, attr=attr)
