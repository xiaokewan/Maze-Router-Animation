from itertools import count
from heapq import heappop, heappush

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


def heuristic(_, a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def a_star_viz(G, source, target, direction_factor=1):
    c = count()
    queue = [(0, next(c), source, [], 0)]  # (priority, current node, path, cost)
    visited = {}  # This will store the distance from source to the node
    visited_priority = {}
    visited_nodes_pr = []  # For visualization

    while queue:
        print(queue)
        (priority, _, current, path, cost) = heappop(queue)
        if current not in visited:
            visited[current] = cost  # Update the cost for the current node
            visited_priority[current] = priority
            visited_nodes_pr.append(visited_priority.copy())  # For visualization
            new_path = path + [current]

            if current == target:
                return new_path, visited, visited_nodes_pr  # Return path and visited dict

            for neighbor, data in G[current].items():
                edge_cost = data.get('length', 1)  # Use 1 as default edge cost if 'length' is not available
                if neighbor not in visited:
                    total_cost = cost + edge_cost
                    priority = total_cost + direction_factor * heuristic(G, neighbor, target)
                    heappush(queue, (priority, next(c), neighbor, new_path, total_cost))

    return None, visited, visited_nodes_pr  # In case no path is found


def elp_route(G, source, target, expected_pathlength=50, in_used_paths=None,  direction_factor=1):
    c = count()
    queue = [(0, next(c), source, [], 0)]  # (priority, count, current node, path, cost)
    visited = {}  # This will now be a dict storing cost from source to each node
    visited_nodes = []  # For visualization, storing copies of the visited dict
    shortest = heuristic(G,source,target)
    in_used_opposite_edges = set()
    in_used_edges = set()
    if in_used_paths:
        for path in in_used_paths:
            for u, v in zip(path, path[1:]):
                in_used_edges.add((u, v))
                # Assuming the handling of opposite edges is no longer needed or is handled differently

    while queue:
        (priority, _, current, path, cost) = heappop(queue)
        # looks can not remove this restriction quickly
        print("pop:", (priority, current, path, cost))
        if current not in visited:
            visited[current] = cost  # Record the cost for reaching current node
            visited_nodes.append(visited.copy())  # For visualization
            new_path = path + [current]

            if current == target and cost == expected_pathlength:
                return new_path, visited, visited_nodes  # Return both the path and the visited dict with costs
            # else:
            for neighbor, data in G[current].items():
                # if (current, neighbor) not in in_used_edges:
                if neighbor not in visited:
                    # print("in_used_edges:", in_used_edges)
                    edge_cost = data.get('length', 1)  # Use 1 as default edge cost if 'length' is not available
                    remaining_cost = expected_pathlength - (cost + edge_cost)
                    total_cost = cost + edge_cost

                    if remaining_cost >= 0:
                        if total_cost + heuristic(G, neighbor, target) <= expected_pathlength:
                            priority = ( expected_pathlength - total_cost - 0.9*heuristic(G, neighbor, target))
                            if priority < 0:
                                priority = 0
                            print("push:",(priority, next(c), neighbor, new_path, total_cost))
                            heappush(queue, (priority, next(c), neighbor, new_path, total_cost))

    return None, visited, visited_nodes  # In case no path is found, return None and the visited dict


def constrained_a_star_viz(G, source, target, expected_pathlength, in_used_paths=None, direction_factor = 1):
    queue = [(0, source, [], 0)]
    visited = set()
    visited_nodes = []
    in_used_opposite_edges = set()
    in_used_edges = set()

    if in_used_paths:
        for path in in_used_paths:
            for u, v in zip(path, path[1:]):
                t_edge = G.edges[u, v]
                in_used_edges.add((u, v))
                # e1_o, e2_o = t_edge['opposite']
                # in_used_opposite_edges.add((e1_o, e2_o))

    while queue:
        (priority, current, path, cost) = heappop(queue)
        # 原版 visited是 set
        # if current not in visited:
        #     visited.add(current)
        #     visited_nodes.append(visited.copy())
        #     new_path = path + [current]

        # 新版 visited是 lib
        if current not in visited:
            visited[current] = cost  # Record the cost for reaching current node
            visited_nodes.append(visited.copy())  # For visualization
            new_path = path + [current]

            if current == target:
                if cost == expected_pathlength:
                    return new_path, visited, visited_nodes
            else:
                for neighbor, data in G[current].items():
                    # Check if the edge is in the in_used_opposite_edges
                    if (current, neighbor) in in_used_opposite_edges or (current, neighbor) in in_used_edges:
                        continue

                    edge_cost = data['length']
                    remaining_cost = expected_pathlength - (cost + edge_cost)

                    # Check if the opposite edge is in the current path
                    t_edge = G.edges[current, neighbor]
                    # e1_o, e2_o = t_edge['opposite']
                    # if (e1_o, e2_o) in zip(path, path[1:]):
                    #     continue

                    if remaining_cost >= 0:
                        if cost + edge_cost + heuristic(G, neighbor, target) <= expected_pathlength:
                            priority = expected_pathlength - (cost + edge_cost + heuristic(G, neighbor, target))
                            heappush(queue, (priority, neighbor, new_path, cost + edge_cost))
    return None, visited, visited_nodes
