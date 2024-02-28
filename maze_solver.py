
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


# grid graph
def create_grid_graph_with_obstacles(dim_x, dim_y, obstacles):
    G = nx.grid_2d_graph(dim_x, dim_y)
    for obstacle in obstacles:
        G.remove_node(obstacle) if obstacle in G else None
    return G

nx.shortest_path
# 执行BFS并跟踪从源点开始的距离
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


# 使用BFS找到路径（从目的地回溯到源点）
def bfs_find_path(parents, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


dim_x, dim_y = 20, 20
obstacles = [(4, 4), (4, 5), (4, 6), (5, 4), (6, 4), (11, 16)]
G = create_grid_graph_with_obstacles(dim_x, dim_y, obstacles)
start, end = (0, 0), (19, 19)
visited = bfs_with_distances(G, start)

# use the nx embedded router, method only can be dijkstra/ bellman-ford
if router == 'dijkstra':
    path = nx.shortest_path(G, start, end, method='dijkstra')


parents = {start: None}
for node in visited:
    for neighbor in G.neighbors(node):
        if neighbor not in parents:
            parents[neighbor] = node
path = bfs_find_path(parents, start, end)


fig, ax = plt.subplots(figsize=(10, 10))
plt.axis('off')


def update(frame):
    if frame > visited[end]:
        frame = visited[end]
    ax.clear()
    plt.axis('off')
    ax.set_xlim(-0.5, dim_x - 0.5)
    ax.set_ylim(-0.5, dim_y - 0.5)

    # draw units of grid
    for x in range(dim_x):
        for y in range(dim_y):
            if (x, y) in obstacles:
                ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="black"))
            elif (x, y) in visited and visited[(x, y)] <= frame:
                color = "skyblue"
                ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color=color))
                ax.text(y, x, str(visited[(x, y)]), ha='center', va='center', color='black')

    # 如果波前到达终点，开始绘制路径
    if frame >= visited[end]:
        for pos in path:
            if visited[pos] <= frame:
                ax.add_patch(Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="yellow"))
                ax.text(pos[1], pos[0], str(visited[pos]), ha='center', va='center', color='black')

    # 标记起点和终点
    ax.add_patch(Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, color="lime"))
    ax.text(start[1], start[0], 'S', ha='center', va='center', color='black')
    ax.add_patch(Rectangle((end[1] - 0.5, end[0] - 0.5), 1, 1, color="red"))
    ax.text(end[1], end[0], 'E', ha='center', va='center', color='black')


extra_frames = 25  # 大约五秒的延迟帧数
ani = FuncAnimation(fig, update, frames=max(visited.values()) + 1 + extra_frames, interval=200, repeat=False)

# 保存动画
ani.save('./bfs_grid_animation_with_path_and_weights.gif', writer='pillow', dpi=140)
