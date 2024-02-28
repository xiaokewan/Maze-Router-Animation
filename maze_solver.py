from matplotlib import pyplot as plt
from itertools import count
from heapq import heappop, heappush
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import Image, display
import os
import networkx as nx
import heapq

def heuristic(_, a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def a_star_viz(G, source, target, direction_factor=1):
    queue = [(0, source, [], 0)]  # (priority, current node, path, cost)
    visited = {}  # This will store the distance from source to the node
    visited_nodes = []  # For visualization

    while queue:
        (priority, current, path, cost) = heappop(queue)
        if current not in visited:
            visited[current] = cost  # Update the cost for the current node
            visited_nodes.append(visited.copy())  # For visualization
            new_path = path + [current]

            if current == target:
                return new_path, visited  # Return path and visited dict

            for neighbor, data in G[current].items():
                edge_cost = data.get('length', 1)  # Use 1 as default edge cost if 'length' is not available
                if neighbor not in visited:
                    total_cost = cost + edge_cost
                    priority = total_cost + direction_factor*heuristic(G, neighbor, target)
                    heappush(queue, (priority, neighbor, new_path, total_cost))

    return None, visited  # In case no path is found


def constrained_a_star_viz(G, source, target, expected_pathlength = 50, in_used_paths=None):
    queue = [(0, source, [], 0)]  # (priority, current node, path, cost)
    visited = {}  # This will now be a dict storing cost from source to each node
    visited_nodes = []  # For visualization, storing copies of the visited dict

    in_used_opposite_edges = set()
    in_used_edges = set()
    if in_used_paths:
        for path in in_used_paths:
            for u, v in zip(path, path[1:]):
                in_used_edges.add((u, v))
                # Assuming the handling of opposite edges is no longer needed or is handled differently

    while queue:
        (priority, current, path, cost) = heapq.heappop(queue)
        # looks can not remove this restriction quickly
        if current not in path:
            visited[current] = cost  # Record the cost for reaching current node
            visited_nodes.append(visited.copy())  # For visualization
            new_path = path + [current]

            if current == target and cost == expected_pathlength:
                return new_path, visited, visited_nodes  # Return both the path and the visited dict with costs
            else:
                for neighbor, data in G[current].items():
                    if (current, neighbor) not in in_used_edges:
                        edge_cost = data.get('length', 1)  # Use 1 as default edge cost if 'length' is not available
                        remaining_cost = expected_pathlength - (cost + edge_cost)
                        total_cost = cost + edge_cost

                        if remaining_cost >= 0:
                            if total_cost + heuristic(G, neighbor, target) <= expected_pathlength:
                                priority = expected_pathlength - total_cost + heuristic(G, neighbor, target)
                                heapq.heappush(queue, (priority, neighbor, new_path, total_cost))

    return None, visited, visited_nodes  # In case no path is found, return None and the visited dict


# grid graph
def create_grid_graph_with_obstacles(dim_x, dim_y, obstacles):
    G = nx.grid_2d_graph(dim_x, dim_y)
    for obstacle in obstacles:
        G.remove_node(obstacle) if obstacle in G else None
    weights = {edge: 1 for edge in G.edges()}
    nx.set_edge_attributes(G, weights, 'length')
    return G


#
def bfs_with_distances(G, start):
    visited = {start: 0}
    queue = [start]
    while queue:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1
                queue.append(neighbor)
    return visited


def bfs_find_path(parents, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def animate_maze_router(start, end, grid_size, obstacles, router='dijkstra', attr={'direction_factor':1, 'expect_pathlength':32}):
    def constraint_update(frame):
        ax.clear()
        plt.axis('off')
        ax.set_xlim(-0.5, dim_x - 0.5)
        ax.set_ylim(-0.5, dim_y - 0.5)

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
        if path and frame == len(visited_nodes) - 1 + extra_frames:
            for pos in path:
                ax.add_patch(Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="yellow"))
                ax.text(pos[1], pos[0], str(current_visited.get(pos, '')), ha='center', va='center', color='black')

    def update(frame):

        max_frame = max(visited.values(), default=0)
        frame_limit = visited.get(end, max_frame)
        frame = min(frame, frame_limit)

        ax.clear()
        plt.axis('off')
        ax.set_xlim(-0.5, dim_x - 0.5)
        ax.set_ylim(-0.5, dim_y - 0.5)

        max_distance = max(visited.values(), default=1)
        for x in range(dim_x):
            for y in range(dim_y):
                if (x, y) in obstacles:
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="black"))
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
                    ax.add_patch(Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="yellow"))
                    ax.text(pos[1], pos[0], str(visited.get(pos, '')), ha='center', va='center', color='black')

        # start and end label
        ax.add_patch(Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, color="lime"))
        ax.text(start[1], start[0], 'SOURCE', ha='center', va='center', color='black')
        ax.add_patch(Rectangle((end[1] - 0.5, end[0] - 0.5), 1, 1, color="red"))
        ax.text(end[1], end[0], 'SINK', ha='center', va='center', color='black')


    dim_x, dim_y = grid_size
    G = create_grid_graph_with_obstacles(dim_x, dim_y, obstacles)
    # use the nx embedded router, method only can be dijkstra/ bellman-ford
    if router == 'dijkstra' or router == 'bellman-ford' or router == 'bfs':
        path = nx.shortest_path(G, start, end, method='dijkstra')  # 注意这里可能需要根据router调整方法
        visited = bfs_with_distances(G, start)
    elif router == 'a_star':
        df = attr["direction_factor"]
        path, visited = a_star_viz(G, start, end, direction_factor=df)
    elif router == 'constrained_a_star':
        ep = attr["expect_pathlength"]
        path, visited, visited_nodes = constrained_a_star_viz(G, start, end, ep)
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
    # if router == "constrained_a_star":
    #     ani = FuncAnimation(fig, constraint_update, frames=len(visited_nodes) + 1 + extra_frames, interval=200, repeat=False)
    #     dpi = 60
    # else:
    ani = FuncAnimation(fig, update, frames=max(visited.values()) + 1 + extra_frames, interval=200, repeat=False)
    dpi = 140
    file_name = './{}_grid_animation_with_path_and_weights.gif'.format(router)
    ani.save(file_name, writer='pillow', dpi = dpi)
    display(Image(filename=file_name))
    file_path = os.path.join(os.getcwd(), file_name)
    os.startfile(file_path)
    print("Success simulation, animation saved in address")


if __name__ == "__main__":
    start, end = (0, 0), (15, 17)
    obstacles = [(4, 4), (4, 5), (4, 6), (5, 4), (6, 4), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18)]
    grid_size = [20, 20]
    router = 'constrained_a_star'
    attr = {'direction_factor': 1.0,    # for a_star
            'expect_pathlength': 36}    # for constrained a_star
    animate_maze_router(start, end, grid_size, obstacles, router, attr=attr)
