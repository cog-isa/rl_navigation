from queue import PriorityQueue


def astar_path(start, goal, get_neighbors,
               heuristic=lambda x, y: 0,
               get_cost=lambda x, y: 1):

    start, goal = tuple(start), tuple(goal)
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()[1]
        if current == goal:
            break

        for next_node in get_neighbors(current):
            new_cost = cost_so_far[current] + get_cost(current, next_node)

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost

                priority = new_cost + heuristic(goal, next_node)
                frontier.put((priority, next_node))
                came_from[next_node] = current

    path = list([])
    while current != start:
        path.append(current)
        current = came_from[current]

    return path[::-1]


def neighbors_from_traversable(traversable):
    def hyp_nbr(x):
        tmp = []
        max_x, max_y = traversable.shape
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (not (i == 0 and j == 0)) and \
                        x[0] + i >= 0 and x[0] + i <= max_x - 1 and \
                        x[1] + j >= 0 and x[1] + j <= max_y - 1:
                    tmp.append((x[0] + i, x[1] + j))

        return tmp

    def f(x):
        if traversable[x[0], x[1]] == False:
            print("This area is not traversable", x)
            return None
        else:
            neighbors = []
            for i in hyp_nbr(x):
                if traversable[i[0], i[1]]:
                    neighbors.append(i)
        return neighbors

    return f