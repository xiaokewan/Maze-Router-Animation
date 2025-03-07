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
    visited = {}
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
        # org visited is set
        # if current not in visited:
        #     visited.add(current)
        #     visited_nodes.append(visited.copy())
        #     new_path = path + [current]

        # now vsisured is lib
        if current not in visited:
            visited[current] = cost  # Record the cost for reaching current node
            visited_nodes.append(visited.copy())  # For visualization
            new_path = path + [current]

            if current == target:
                if cost == expected_pathlength:
                    return new_path, visited, visited_nodes
                else:
                    continue
            else:
                for neighbor, data in G[current].items():
                    # Check if the edge is in the in_used_opposite_edges
                    if (current, neighbor) in in_used_opposite_edges or (current, neighbor) in in_used_edges:
                        continue

                    edge_cost = data['length']
                    # remaining_cost = expected_pathlength - (cost + edge_cost)
                    gn = edge_cost + cost
                    hn = heuristic(G, neighbor, target)
                    fn = expected_pathlength - hn - gn
                    # Check if the opposite edge is in the current path
                    # t_edge = G.edges[current, neighbor]
                    # e1_o, e2_o = t_edge['opposite']
                    # if (e1_o, e2_o) in zip(path, path[1:]):
                    #     continue

                    if fn >= 0:
                        priority = expected_pathlength - (cost + edge_cost + heuristic(G, neighbor, target))
                        heappush(queue, (priority, neighbor, new_path, cost + edge_cost))
    return None, visited, visited_nodes


def lm(G, source, target, expected_pathlength, in_used_paths=None, PIC=False):
    # PIC stands for the restriction of PIC
    c = count()
    queue = [(0, next(c), source, [], 0)]  # (priority, count, current node, path, cost)
    visited = {}  # This will now be a dict storing cost from source to each node
    visited_nodes = []  # For visualization, storing copies of the visited dict
    in_used_opposite_edges = set()

    in_used_edges = set()
    tt_pushed = 1 # track the total pushed nodes

    if in_used_paths:
        for path in in_used_paths:
            # as the in_used_paths shown as: [length,[path]]
            for u, v in zip(path[1], path[1][1:]):
                t_edge = G.edges[u, v]
                in_used_edges.add((u, v))
                if PIC:
                    e1_o, e2_o = t_edge['opposite']
                    in_used_opposite_edges.add((e1_o, e2_o))

    while queue:
        (priority, _, current, path, cost) = heappop(queue)
        if current not in visited or priority <= visited[current]:
            visited[current] = priority  # Record the cost for reaching current node
            visited_nodes.append(visited.copy()) # For visualization
            new_path = path + [current]

            if current == target:
                if cost == expected_pathlength:
                    return new_path, visited, visited_nodes, tt_pushed # Return both the path and the visited dict with costs
                else:
                    continue
            for neighbor, data in G[current].items():
                if ((current, neighbor) in in_used_opposite_edges or (current, neighbor) in in_used_edges
                        or (current, neighbor) in set(zip(new_path, new_path[1:])) | set(zip(new_path[1:], new_path)))\
                        or neighbor in new_path:
                    continue

                ecost = data.get('length', 1)  # Use 1 as default edge cost if 'length' is not available
                gn = cost + ecost
                hn = heuristic(G, neighbor, target)
                # hn = 0
                fn = expected_pathlength-gn-hn

                # opposite restriction of RC-PIC

                if PIC:
                    if (G.edges[current, neighbor]['opposite']) in zip(path, path[1:]):
                        continue
                if fn >= 0 and (neighbor not in visited or fn <= visited[neighbor]):
                    # fn is the priority of the queue
                    heappush(queue, (fn, next(c), neighbor, new_path, gn))
                    tt_pushed += 1
    return None, visited, visited_nodes, tt_pushed




def lm_max_f(G, source, target, L, in_used_paths=None, PIC=False):
    c = count()
    queue = [(0, next(c), source, [], 0)]
    visited = {}
    visited_nodes = []
    in_used_opposite_edges = set()
    in_used_edges = set()
    tt_pushed = 1

    if in_used_paths:
        for path in in_used_paths:
            for u, v in zip(path[1], path[1][1:]):
                t_edge = G.edges[u, v]
                in_used_edges.add((u, v))
                if PIC:
                    e1_o, e2_o = t_edge['opposite']
                    in_used_opposite_edges.add((e1_o, e2_o))

    while queue:
        (priority, _, current, path, g) = heappop(queue)
        priority = -priority  # Re-invert back to normal

        if current not in visited or priority <= visited[current]:
            visited[current] = g  # Record the best f(n)
            visited_nodes.append(visited.copy())  # For visualization
            new_path = path + [current]

            if current == target:
                if g == L:
                    print(f" final_path: {new_path}, pushed amount: {tt_pushed}")
                    return new_path, visited, visited_nodes, tt_pushed  # Return valid path
                else:
                    continue

            for neighbor, data in G[current].items():
                if ((current, neighbor) in in_used_opposite_edges or (current, neighbor) in in_used_edges
                    or (current, neighbor) in set(zip(new_path, new_path[1:])) | set(zip(new_path[1:], new_path))) \
                        or neighbor in new_path:
                    continue

                ecost = data.get('length', 1)  # Default edge cost = 1 if 'length' is not available
                gn = g + ecost
                hn = heuristic(G, neighbor, target)
                fn = L - gn - hn  # f(n) calculation
                # fn = gn + hn
                if PIC:
                    if (G.edges[current, neighbor]['opposite']) in zip(path, path[1:]):
                        continue

                if fn >= 0 and (neighbor not in visited or fn <= visited[neighbor]):
                    print(f"search a node {-fn, next(c), neighbor, new_path, gn}")
                    heappush(queue, (-fn, next(c), neighbor, new_path, gn))
                    tt_pushed += 1

    return None, visited, visited_nodes, tt_pushed


def lm_dual_stage_v2(G, s, t, L, in_use=None, PIC=False):
    """
    Two-stage LM Algorithm:
    1. Stage 1: Finds the shortest path using A*-like search.
    2. Stage 2: If the shortest path is too short, reuses the queue and modifies priorities to maximize path length.
    """

    c = count()
    q = [(0, next(c), s, [], 0)]  # (priority, count, current node (bu), path (pt), cost)
    vis = {}  # Stores shortest cost to each node
    vis_n = []  # Stores visited nodes for visualization
    tp = 1  # Total pushed nodes tracker

    iu_opp = set()
    iu_edges = set()

    if in_use:
        for p in in_use:
            for u, v in zip(p[1], p[1][1:]):
                t_edge = G.edges[u, v]
                iu_edges.add((u, v))
                if PIC:
                    e1_o, e2_o = t_edge['opposite']
                    iu_opp.add((e1_o, e2_o))

    # ---- STAGE 1: Shortest Path Search (Standard A*) ----
    spaths = []
    scost = float('inf')

    while q:
        prio, _, bu, pt, g = heappop(q)

        if bu in vis and g > vis[bu]:  # Allow equal cost paths
            continue

        vis[bu] = g  # Record shortest path cost
        vis_n.append(vis.copy())
        npt = pt + [bu]

        if bu == t:
            if scost == float('inf'):
                scost = g  # Set shortest cost once we find the first shortest path

            if g == scost:  # Store unique shortest paths
                spaths.append(npt)
                heappush(q, (prio, next(c), bu, npt, g))
                break

        for v, d in G[bu].items():
            if ((bu, v) in iu_opp or (bu, v) in iu_edges
                or (bu, v) in set(zip(npt, npt[1:])) | set(zip(npt[1:], npt))) \
                    or v in npt:
                continue

            ec = d.get('length', 1)  # Default edge cost if not provided
            gn = g + ec
            hn = heuristic(G, v, t)
            fn = gn + hn  # A* priority function for shortest path

            heappush(q, (fn, next(c), v, npt, gn))
            tp += 1

    # If the shortest path is already long enough, return it
    if scost >= L:
        return spaths[0], vis, vis_n, tp

    # ---- STAGE 2: Detour Search (Modify Priorities) ----
    nq = []
    nvis = vis.copy()  # Transfer visited data to prevent re-exploring nodes

    while q:
        _, cnt, bu, pt, g = heappop(q)
        fn = L - g
        heappush(nq, (fn, cnt, bu, pt, g))

    q = nq
    vis = nvis

    while q:
        prio, _, bu, pt, g = heappop(q)

        if bu in vis and prio > L - (vis[bu] + heuristic(G, bu, t)):  # Avoid re-exploring processed nodes
            continue

        vis[bu] = g
        vis_n.append(vis.copy())
        npt = pt + [bu]

        if bu == t and g == L:
            print(f"Found path: {npt}, \nTotal pushed nodes: {tp}")
            return npt, vis, vis_n, tp

        for v, d in G[bu].items():
            if ((bu, v) in iu_opp or (bu, v) in iu_edges
                or (bu, v) in set(zip(npt, npt[1:])) | set(zip(npt[1:], npt))) \
                    or v in npt:
                continue

            ec = d.get('length', 1)
            gn = g + ec
            hn = heuristic(G, v, t)
            fn = L - gn - hn
            if fn < 0 or gn > L:
                continue
            heappush(q, (fn, next(c), v, npt, gn))
            tp += 1

    return None, vis, vis_n, tp  # No valid path found



def lm_dual_stage(G, source, target, expected_pathlength, in_used_paths=None, PIC=False):
    """
    Two-stage LM Algorithm:
    1. Stage 1: Finds the shortest path using A*-like search.
    2. Stage 2: If the shortest path is too short, reuses the queue and modifies priorities to maximize path length.
    """

    c = count()
    queue = [(0, next(c), source, [], 0)]  # (priority, count, current node, path, cost)
    visited = {}  # Stores shortest cost to each node
    visited_nodes = []
    tt_pushed = 1  # Total pushed nodes tracker

    in_used_opposite_edges = set()
    in_used_edges = set()

    if in_used_paths:
        for path in in_used_paths:
            for u, v in zip(path[1], path[1][1:]):
                t_edge = G.edges[u, v]
                in_used_edges.add((u, v))
                if PIC:
                    e1_o, e2_o = t_edge['opposite']
                    in_used_opposite_edges.add((e1_o, e2_o))

    # ---- STAGE 1: Shortest Path Search (Standard A*) ----
    shortest_paths = []
    shortest_cost = float('inf')
    wavefront_queue = []  # Stores the frontier for Stage 2

    while queue:
        priority, _, current, path, cost = heappop(queue)

        if current in visited and cost > visited[current]:  # Allow equal cost paths
            continue

        visited[current] = cost  # Record shortest path cost
        visited_nodes.append(visited.copy())
        new_path = path + [current]

        if current == target:
            if shortest_cost == float('inf'):
                shortest_cost = cost  # Set shortest cost once we find the first shortest path

            if cost == shortest_cost:  # Store unique shortest paths
                shortest_paths.append(new_path)
            else:
                # Stop once a longer path appears
                heappush(wavefront_queue, (cost, next(c), current, new_path, cost))
                break

        for neighbor, data in G[current].items():
            if ((current, neighbor) in in_used_opposite_edges or (current, neighbor) in in_used_edges
                or (current, neighbor) in set(zip(new_path, new_path[1:])) | set(zip(new_path[1:], new_path))) \
                    or neighbor in new_path:
                continue

            ecost = data.get('length', 1)  # Default edge cost if not provided
            gn = cost + ecost
            hn = heuristic(G, neighbor, target)
            fn = gn + hn  # A* priority function for shortest path

            heappush(queue, (fn, next(c), neighbor, new_path, gn))
            tt_pushed += 1

    # If the shortest path is already long enough, return it
    if shortest_cost >= expected_pathlength:
        return shortest_paths[0], visited, visited_nodes, tt_pushed

    # ---- STAGE 2: Detour Search (Reusing the Queue Correctly) ----
    queue = wavefront_queue  # Directly use the frontier from Stage 1
    visited = visited.copy()  # Transfer visited data

    while queue:
        _, _, current, path, cost = heappop(queue)

        if current in visited and cost > visited[current]:  # Avoid re-exploring processed nodes
            continue

        visited[current] = cost  # Record new path cost
        visited_nodes.append(visited.copy())
        new_path = path + [current]

        if current == target and cost == expected_pathlength:
            print(f"Found the path: {new_path}, \n with total pushed nodes: {tt_pushed}")
            return new_path, visited, visited_nodes, tt_pushed  # Return exact-length path

        for neighbor, data in G[current].items():
            if ((current, neighbor) in in_used_opposite_edges or (current, neighbor) in in_used_edges
                or (current, neighbor) in set(zip(new_path, new_path[1:])) | set(zip(new_path[1:], new_path))) \
                    or neighbor in new_path:
                continue

            ecost = data.get('length', 1)
            gn = cost + ecost
            hn = heuristic(G, neighbor, target)
            fn = expected_pathlength - gn - hn  # Detour search priority function

            if fn < 0 or gn > expected_pathlength:  # Pruning unhelpful paths
                continue

            heappush(queue, (fn, next(c), neighbor, new_path, gn))
            tt_pushed += 1

    return None, visited, visited_nodes, tt_pushed  # No valid path found
