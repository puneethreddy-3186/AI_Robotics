# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.entry_count = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        pop = heapq.heappop(self.queue)
        return pop[0], pop[2]

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        self.queue.remove(node)
        return self.queue

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        'Add a new task or update the priority of an existing task'
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        heapq.heappush(self.queue, (node[0], self.entry_count, node[1]))
        self.entry_count += 1

    def is_in_frontier(self, key):
        return key in [n[-1][-1] for n in self.queue]

    def find_lowest_cost_neighbour(self, key):
        lowest_cost = float('inf')
        lowest_cost_neighbour = None
        for n in self.queue:
            if key == n[-1][-1] and n[0] < lowest_cost:
                lowest_cost = n[0]
                lowest_cost_neighbour = n
        return lowest_cost_neighbour

    def find_lowest_cost_path_with_node(self, key):
        lowest_cost = float('inf')
        lowest_cost_path = None
        for n in self.queue:
            if key in n[-1] and n[0] < lowest_cost:
                lowest_cost = n[0]
                lowest_cost_path = n
        return lowest_cost_path

    def find_lowest_path(self):
        lowest_cost = float('inf')
        lowest_cost_path = None
        for n in self.queue:
            if n[0] < lowest_cost:
                lowest_cost = n[0]
                lowest_cost_path = n
        return lowest_cost_path

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    if start == goal:
        return []
    explored_nodes = set()
    frontier = PriorityQueue()
    frontier.append((1, [start]))
    while frontier.size() != 0:
        path = frontier.pop()[1]
        current_node = path[-1]
        explored_nodes.add(current_node)
        sorted_neighbours = sorted(graph[current_node])
        for neighbour in sorted_neighbours:
            if neighbour not in explored_nodes and not (frontier.is_in_frontier(neighbour)):
                new_path = path + [neighbour]
                if neighbour == goal:
                    return new_path
                else:
                    frontier.append((1, new_path))

    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    if start == goal:
        return []
    explored_nodes = set()
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    while frontier.size() != 0:
        priority_path = frontier.pop()
        cost = priority_path[0]
        path = priority_path[1]
        current_node = path[-1]
        explored_nodes.add(current_node)
        if current_node == goal:
            return path
        neighbours = graph[current_node]
        for neighbour in neighbours:
            edge_cost = graph.get_edge_weight(current_node, neighbour)
            new_path = path + [neighbour]
            new_path_cost = cost + edge_cost
            if neighbour not in explored_nodes and not (frontier.is_in_frontier(neighbour)):
                frontier.append((new_path_cost, new_path))
            else:
                lowest_cost_path = frontier.find_lowest_cost_neighbour(neighbour)
                if frontier.is_in_frontier(neighbour) and lowest_cost_path is not None \
                        and (new_path_cost < lowest_cost_path[0]):
                    frontier.remove(lowest_cost_path)
                    frontier.append((new_path_cost, new_path))

    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    v_coord = graph.nodes[v]['pos']
    g_coord = graph.nodes[goal]['pos']
    dist = math.sqrt((v_coord[0] - g_coord[0]) ** 2 + (v_coord[1] - g_coord[1]) ** 2)
    return dist
    raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    if start == goal:
        return []
    explored_nodes = set()
    frontier = PriorityQueue()
    frontier.append((heuristic(graph, start, goal), [start]))
    while frontier.size() != 0:
        priority_path = frontier.pop()
        cost = priority_path[0]
        path = priority_path[1]
        current_node = path[-1]
        explored_nodes.add(current_node)
        if current_node == goal:
            return path
        neighbours = graph[current_node]
        for neighbour in neighbours:
            edge_cost = graph.get_edge_weight(current_node, neighbour) + (cost - heuristic(graph, current_node, goal))
            new_path = path + [neighbour]
            new_path_cost = edge_cost + heuristic(graph, neighbour, goal)
            if neighbour not in explored_nodes and not (frontier.is_in_frontier(neighbour)):
                frontier.append((new_path_cost, new_path))
            else:
                lowest_cost_path = frontier.find_lowest_cost_neighbour(neighbour)
                if frontier.is_in_frontier(neighbour) and lowest_cost_path is not None \
                        and (new_path_cost < lowest_cost_path[0]):
                    frontier.remove(lowest_cost_path)
                    frontier.append((new_path_cost, new_path))

    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def can_terminate(current_node, explored_node_set):
    return current_node in explored_node_set


def calculate_bi_a_star_path(graph, current_path, start, goal, frontier, other_frontier, explored_node_set,
                             other_node_set,
                             heuristic=euclidean_dist_heuristic):
    node_set = set()
    node_set.update(explored_node_set)
    other_set = set()
    other_set.update(other_node_set)
    node_set.add(current_path[1][-1])
    other_frontier_node = other_frontier.find_lowest_path()[-1][-1]
    other_set.add(other_frontier_node)
    intersecting_nodes = node_set.intersection(other_set)
    total_cost = float('inf')
    final_path = None
    for node in sorted(intersecting_nodes):
        path = frontier.find_lowest_cost_path_with_node(node)
        if path is None and node in current_path[1]:
            path = ([current_path[0], 1, current_path[1]])
        other_path = other_frontier.find_lowest_cost_path_with_node(node)
        if other_path is None and node in current_path[1]:
            other_path = (current_path[0], 1, current_path[1])
        if path is not None and other_path is not None:
            cost1, sub_path1 = calculate_a_star_path_cost(graph, start, goal, path, node, euclidean_dist_heuristic)
            cost2, sub_path2 = calculate_a_star_path_cost(graph, start, goal, other_path, node,
                                                          euclidean_dist_heuristic)
            if (cost1 + cost2) < total_cost:
                total_cost = (cost1 + cost2)
                if sub_path1[0] == goal:
                    sub_path1_rev = sub_path1[::-1]
                    final_path = sub_path2 + sub_path1_rev[1:]
                if sub_path2[0] == goal:
                    sub_path2_rev = sub_path2[::-1]
                    final_path = sub_path1 + sub_path2_rev[1:]
    return final_path


def calculate_bi_ucs_path(graph, current_path, goal, frontier, other_frontier, explored_node_set, other_node_set):
    node_set = set()
    node_set.update(explored_node_set)
    other_set = set()
    other_set.update(other_node_set)
    node_set.add(current_path[1][-1])
    lowest_path = other_frontier.find_lowest_path()
    if lowest_path is not None:
        other_frontier_node = other_frontier.find_lowest_path()[-1][-1]
        other_set.add(other_frontier_node)
    intersecting_nodes = node_set.intersection(other_set)
    total_cost = float('inf')
    final_path = None
    for node in sorted(intersecting_nodes):
        path = frontier.find_lowest_cost_path_with_node(node)
        if path is None and node in current_path[1]:
            path = ([current_path[0], 1, current_path[1]])
        other_path = other_frontier.find_lowest_cost_path_with_node(node)
        if other_path is None and node in current_path[1]:
            other_path = (current_path[0], 1, current_path[1])
        if path is not None and other_path is not None:
            cost1, sub_path1 = calculate_path_cost(graph, path, node)
            cost2, sub_path2 = calculate_path_cost(graph, other_path, node)
            if (cost1 + cost2) < total_cost:
                total_cost = (cost1 + cost2)
                if sub_path1[0] == goal:
                    sub_path1_rev = sub_path1[::-1]
                    final_path = sub_path2 + sub_path1_rev[1:]
                if sub_path2[0] == goal:
                    sub_path2_rev = sub_path2[::-1]
                    final_path = sub_path1 + sub_path2_rev[1:]
    return final_path


def calculate_a_star_path_cost(graph, start, goal, path, intersecting_node, heuristic=euclidean_dist_heuristic):
    if path[2][-1] == intersecting_node:
        cost = path[0]
        new_path = path[2]
        h_cost = 0
        if path[2][0] == start:
            h_cost = heuristic(graph, intersecting_node, goal)
        if path[2][0] == goal:
            h_cost = heuristic(graph, intersecting_node, start)
        cost -= h_cost
    else:
        next_node = path[2][path[2].index(intersecting_node) + 1]
        cost = path[0]
        new_path = path[2]
        if next_node is not None:
            h_cost = 0
            if path[2][0] == start:
                h_cost = heuristic(graph, next_node, goal)
            if path[2][0] == goal:
                h_cost = heuristic(graph, next_node, start)
            cost = path[0] - h_cost - graph.get_edge_weight(intersecting_node, next_node)
            new_path = path[2][0:path[2].index(intersecting_node) + 1]
    return cost, new_path


def calculate_path_cost(graph, path, intersecting_node):
    if path[2][-1] == intersecting_node:
        cost = path[0]
        new_path = path[2]
    else:
        next_node = path[2][path[2].index(intersecting_node) + 1]
        cost = path[0]
        new_path = path[2]
        if next_node is not None:
            cost = path[0] - graph.get_edge_weight(intersecting_node, next_node)
            new_path = path[2][0:path[2].index(intersecting_node) + 1]
    return cost, new_path


def process_bi_ucs_node(graph, goal, frontier, other_frontier, explored_node_set, other_node_set):
    priority_path = frontier.pop()
    cost = priority_path[0]
    path = priority_path[1]
    current_node = path[-1]
    if can_terminate(current_node, other_node_set):
        return calculate_bi_ucs_path(graph, priority_path, goal, frontier, other_frontier, explored_node_set,
                                     other_node_set)
    explored_node_set.add(current_node)
    neighbours = graph[current_node]
    for neighbour in neighbours:
        edge_cost = graph.get_edge_weight(current_node, neighbour)
        new_path = path + [neighbour]
        new_path_cost = cost + edge_cost
        if neighbour not in explored_node_set and not (frontier.is_in_frontier(neighbour)):
            frontier.append((new_path_cost, new_path))
        else:
            lowest_cost_path = frontier.find_lowest_cost_neighbour(neighbour)
            if frontier.is_in_frontier(neighbour) and lowest_cost_path is not None \
                    and (new_path_cost < lowest_cost_path[0]):
                frontier.remove(lowest_cost_path)
                frontier.append((new_path_cost, new_path))
    return None


def bidirectional_ucs(graph, start, goal):
    if start == goal:
        return []
    s_explored_nodes = set()
    g_explored_nodes = set()
    s_frontier = PriorityQueue()
    g_frontier = PriorityQueue()
    s_frontier.append((0, [start]))
    g_frontier.append((0, [goal]))
    direction = True
    while s_frontier.size() != 0 and g_frontier.size() != 0:
        if direction:
            path = process_bi_ucs_node(graph, goal, s_frontier, g_frontier, s_explored_nodes, g_explored_nodes)
            direction = False
            if path is not None:
                return path
        else:
            path = process_bi_ucs_node(graph, goal, g_frontier, s_frontier, g_explored_nodes, s_explored_nodes)
            direction = True
            if path is not None:
                return path

    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def process_bi_a_star_node(graph, start, goal, start_node, frontier, other_frontier, explored_node_set, other_node_set,
                           heuristic=euclidean_dist_heuristic):
    priority_path = frontier.pop()
    cost = priority_path[0]
    path = priority_path[1]
    current_node = path[-1]
    if can_terminate(current_node, other_node_set):
        return calculate_bi_a_star_path(graph, priority_path, start, goal, frontier, other_frontier, explored_node_set,
                                        other_node_set, euclidean_dist_heuristic)
    explored_node_set.add(current_node)
    neighbours = graph[current_node]
    for neighbour in neighbours:
        edge_cost = graph.get_edge_weight(current_node, neighbour) + (cost - heuristic(graph, current_node, start_node))
        new_path = path + [neighbour]
        new_path_cost = edge_cost + heuristic(graph, neighbour, start_node)
        if neighbour not in explored_node_set and not (frontier.is_in_frontier(neighbour)):
            frontier.append((new_path_cost, new_path))
        else:
            lowest_cost_path = frontier.find_lowest_cost_neighbour(neighbour)
            if frontier.is_in_frontier(neighbour) and lowest_cost_path is not None \
                    and (new_path_cost < lowest_cost_path[0]):
                frontier.remove(lowest_cost_path)
                frontier.append((new_path_cost, new_path))
    return None


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    if start == goal:
        return []
    s_explored_nodes = set()
    g_explored_nodes = set()
    s_frontier = PriorityQueue()
    g_frontier = PriorityQueue()
    s_frontier.append((heuristic(graph, start, goal), [start]))
    g_frontier.append((heuristic(graph, start, goal), [goal]))
    direction = True
    while s_frontier.size() != 0 and g_frontier.size() != 0:
        if direction:
            path = process_bi_a_star_node(graph, start, goal, goal, s_frontier, g_frontier, s_explored_nodes,
                                          g_explored_nodes,
                                          euclidean_dist_heuristic)
            direction = False
            if path is not None:
                return path
        else:
            path = process_bi_a_star_node(graph, start, goal, start, g_frontier, s_frontier, g_explored_nodes,
                                          s_explored_nodes,
                                          euclidean_dist_heuristic)
            direction = True
            if path is not None:
                return path

    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def can_terminate_tri(current_node, goals, other_node_set_dict):
    found_in = None
    for goal in goals:
        if current_node in other_node_set_dict[goal]:
            found_in = goal
    return found_in


def validate_path(goal_path, found_paths, all_goals):
    goal_nodes = ','.join(sorted([i for i in all_goals if i in goal_path]))
    if goal_nodes not in found_paths:
        found_paths[goal_nodes] = goal_path


def process_tri_ucs_node(graph, start, goals, frontiers_dict, explored_node_dict, found_paths):
    if frontiers_dict[start].size() != 0:
        priority_path = frontiers_dict[start].pop()
        cost = priority_path[0]
        path = priority_path[1]
        current_node = path[-1]
        found_in = can_terminate_tri(current_node, goals, explored_node_dict)
        if found_in is not None:
            goal_path = calculate_bi_ucs_path(graph, priority_path, found_in, frontiers_dict[start],
                                              frontiers_dict[found_in], explored_node_dict[start],
                                              explored_node_dict[found_in])
            all_goals = list(goals)
            all_goals.append(start)
            if goal_path is not None:
                validate_path(goal_path, found_paths, all_goals)
        explored_node_dict[start].add(current_node)
        neighbours = graph[current_node]
        for neighbour in neighbours:
            edge_cost = graph.get_edge_weight(current_node, neighbour)
            new_path = path + [neighbour]
            new_path_cost = cost + edge_cost
            if neighbour not in explored_node_dict[start] and not (frontiers_dict[start].is_in_frontier(neighbour)):
                frontiers_dict[start].append((new_path_cost, new_path))
            else:
                lowest_cost_path = frontiers_dict[start].find_lowest_cost_neighbour(neighbour)
                if frontiers_dict[start].is_in_frontier(neighbour) and lowest_cost_path is not None \
                        and (new_path_cost < lowest_cost_path[0]):
                    frontiers_dict[start].remove(lowest_cost_path)
                    frontiers_dict[start].append((new_path_cost, new_path))
    return None


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goals[0] == goals[1] == goals[2]:
        return []
    goal_set = set(goals)
    explored_nodes_dict = dict()
    frontier_dict = dict()
    found_paths = dict()
    for goal in goal_set:
        explored_nodes_dict[goal] = set()
        frontier_dict[goal] = PriorityQueue()
        frontier_dict[goal].append((0, [goal]))
    found = False
    while not found:
        for goal in sorted(goal_set):
            if len(found_paths) == (len(goal_set) - 1):
                found = True
                break
            process_tri_ucs_node(graph, goal, goal_set.difference(goal), frontier_dict, explored_nodes_dict,
                                 found_paths)
    return stitch_final_path(graph, found_paths, goals, frontier_dict)


def stitch_final_path(graph, found_paths, goals, frontier_dict):
    final_path = None
    all_paths = []
    for path in found_paths.values():
        found_goal_nodes = set([i for i in goals if i in path])
        if len(found_goal_nodes) == 3:
            final_path = path
            break
    if final_path is None:
        paths = list(found_paths.values())
        third_path = find_third_path(graph, found_paths, goals, frontier_dict)
        all_paths.append(stitch_path(paths[0], paths[1]))
        if third_path is not None:
            all_paths.append(stitch_path(paths[0], third_path))
            all_paths.append(stitch_path(paths[1], third_path))
    total_weight = float('inf')
    for p in all_paths:
        try:
            weight = sum_weight(graph, p)
            if weight < total_weight:
                total_weight = weight
                final_path = p
        except Exception as error:
            print(error)
    return final_path


def stitch_final_path_1(graph, found_paths, goals, frontier_dict):
    final_path = None
    all_paths = dict()
    for path in found_paths.values():
        found_goal_nodes = set([i for i in goals if i in path])
        if len(found_goal_nodes) == 3:
            final_path = path
            break
    if final_path is None:
        paths = list(found_paths.values())
        third_path = find_third_path(graph, found_paths, goals, frontier_dict)
        paths.append(third_path)
        path_len1 = sum_weight(graph, paths[0])
        path_len2 = sum_weight(graph, paths[1])
        path_len3 = sum_weight(graph, third_path)
        all_paths[(path_len1 + path_len2)] = stitch_path(paths[0], paths[1])
        all_paths[(path_len1 + path_len3)] = stitch_path(paths[0], paths[2])
        all_paths[(path_len2 + path_len3)] = stitch_path(paths[1], paths[2])
        all_paths = sorted(all_paths.items(), key=lambda item: item[0])
        final_path = all_paths[0][1]
    return final_path


def sum_weight(graph, path):
    """
    Calculate the total cost of a path by summing edge weights.

    Args:
        graph (ExplorableGraph): Graph that contains path.
        path (list(nodes)): List of nodes from src to dst.

    Returns:
        Sum of edge weights in path.
    """
    pairs = zip(path, path[1:])

    return sum([graph.get_edge_data(a, b)['weight'] for a, b in pairs])


def stitch_path(path1, path2):
    intersecting_node = find_intersection([path1, path2])
    if path1[-1] != intersecting_node:
        path1.reverse()
    if path2[0] != intersecting_node:
        path2.reverse()
    return path1 + path2[1:]


def find_third_path(graph, found_paths, goals, frontier_dict):
    missed_goals = list()
    for path in found_paths:
        for i in goals:
            if i not in path:
                missed_goals.append(i)
    sorted(missed_goals)
    'check both frontiers for the path'
    for g in missed_goals:
        pq = frontier_dict[g]
        for entry in pq.queue:
            if all(x in entry for x in goals):
                return entry
    'pull the frontier nodes of one of the pq'
    pq = frontier_dict[missed_goals[0]]
    f_nodes = {entry[2][-1]: entry for entry in pq.queue}
    other_pq = frontier_dict[missed_goals[1]]
    lowest_path_cost = float('inf')
    third_path = None
    for node, value in f_nodes.items():
        possible_paths = [entry for entry in other_pq.queue if node in entry[2]]
        for p in possible_paths:
            sub_p = p[2][0:p[2].index(node) + 1]
            merged_path = stitch_path(value[2], sub_p)
            weight = sum_weight(graph, merged_path)
            if weight < lowest_path_cost:
                lowest_path_cost = weight
                third_path = merged_path
    return third_path


def find_intersection(paths):
    intersecting_node_set = set(paths[0]).intersection(set(paths[1]))
    if len(intersecting_node_set) == 1:
        return intersecting_node_set.pop()
    else:
        for node in intersecting_node_set:
            if node == paths[0][0] or node == paths[0][-1]:
                return node


def process_tri_a_star_node(graph, start, goal, other_goals, frontiers_dict, explored_node_dict, found_paths,
                            heuristic=euclidean_dist_heuristic):
    if frontiers_dict[start].size() != 0:
        priority_path = frontiers_dict[start].pop()
        cost = priority_path[0]
        path = priority_path[1]
        current_node = path[-1]
        found_in = can_terminate_tri(current_node, other_goals, explored_node_dict)
        if found_in is not None:
            goal_path = calculate_bi_ucs_path(graph, priority_path, found_in, frontiers_dict[start],
                                              frontiers_dict[found_in], explored_node_dict[start],
                                              explored_node_dict[found_in])
            all_goals = list(other_goals)
            all_goals.append(start)
            if goal_path is not None:
                validate_path(goal_path, found_paths, all_goals)
        explored_node_dict[start].add(current_node)
        neighbours = graph[current_node]
        for neighbour in neighbours:
            edge_cost = graph.get_edge_weight(current_node, neighbour) + (
                    cost - heuristic(graph, current_node, goal))
            new_path = path + [neighbour]
            new_path_cost = edge_cost + heuristic(graph, neighbour, goal)
            if neighbour not in explored_node_dict[start] and not (frontiers_dict[start].is_in_frontier(neighbour)):
                frontiers_dict[start].append((new_path_cost, new_path))
            else:
                lowest_cost_path = frontiers_dict[start].find_lowest_cost_neighbour(neighbour)
                if frontiers_dict[start].is_in_frontier(neighbour) and lowest_cost_path is not None \
                        and (new_path_cost < lowest_cost_path[0]):
                    frontiers_dict[start].remove(lowest_cost_path)
                    frontiers_dict[start].append((new_path_cost, new_path))
    return None


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goals[0] == goals[1] == goals[2]:
        return []
    result_goals = []
    [result_goals.append(item) for item in goals if item not in result_goals]
    explored_nodes_dict = dict()
    frontier_dict = dict()
    found_paths = dict()
    opposite_goal_dict = dict()
    for goal in result_goals:
        explored_nodes_dict[goal] = set()
        frontier_dict[goal] = PriorityQueue()
        opposite_goal_dict[goal] = find_opposite_goal(goal, result_goals)
        # opposite_goal_dict[goal] = goal_set.difference(goal).pop()
        frontier_dict[goal].append((heuristic(graph, goal, opposite_goal_dict[goal]), [goal]))
    found = False
    while not found:
        for goal in goals:
            if len(found_paths) == (len(result_goals) - 1):
                found = True
                break
            other_goals = [item for item in result_goals if item != goal]
            process_tri_a_star_node(graph, goal, opposite_goal_dict[goal], other_goals, frontier_dict,
                                    explored_nodes_dict,
                                    found_paths, heuristic)
    return stitch_final_path(graph, found_paths, goals, frontier_dict)
    raise NotImplementedError


def find_opposite_goal(start, all_goals):
    return all_goals[(all_goals.index(start) + 1) % len(all_goals)]


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Puneeth Reddy"
    raise NotImplementedError


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    return bidirectional_a_star(graph, start, goal)
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    constOutFront = 2 * 6371  # Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0] - vLatLong[0]) / 2)) ** 2  # First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0]) * math.cos(goalLatLong[0]) * (
            (math.sin((goalLatLong[1] - vLatLong[1]) / 2)) ** 2)  # Second term
    return constOutFront * math.asin(math.sqrt(term1InSqrt + term2InSqrt))  # Straight application of formula
