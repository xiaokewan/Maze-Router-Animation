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


def animate_maze_router(start, end, grid_size, obstacles, router='dijkstras'):

    def update(frame):
        if frame > visited[end]:
            frame = visited[end]
        ax.clear()
        plt.axis('off')
        ax.set_xlim(-0.5, dim_x - 0.5)
        ax.set_ylim(-0.5, dim_y - 0.5)

        max_distance = max(visited.values())  # 获取最大距离用于归一化
        for x in range(dim_x):
            for y in range(dim_y):
                if (x, y) in obstacles:
                    ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color="black"))
                elif (x, y) in visited and visited[(x, y)] <= frame:
                    # 根据距离计算颜色
                    distance = visited[(x, y)]
                    # 归一化距离值，使其在0到1之间
                    normalized_distance = distance / max_distance
                    # 生成颜色，这里使用蓝色的不同深浅来表示距离，越远越深
                    color = plt.cm.Blues(normalized_distance + 0.1)  # +0.1确保即使是最近的也不会是完全白色
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

    dim_x, dim_y = grid_size

    G = create_grid_graph_with_obstacles(dim_x, dim_y, obstacles)

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
    extra_frames = 25  # 大约五秒的延迟帧数
    ani = FuncAnimation(fig, update, frames=max(visited.values()) + 1 + extra_frames, interval=200, repeat=False)
    ani.save('./bfs_grid_animation_with_path_and_weights.gif', writer='pillow', dpi=140)

    print("Success simulation, animation saved in address")


if __name__ == "__main__":
    start, end = (0, 0), (19, 19)
    obstacles = [(4, 4), (4, 5), (4, 6), (5, 4), (6, 4), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19)]
    grid_size = [20, 20]

    animate_maze_router(start, end, grid_size, obstacles)
