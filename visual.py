import random
import heapq
from collections import deque
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap


# loading maze from text file
def load_maze(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    maze = [list(line.strip()) for line in lines if
            line.strip() and not line.startswith("start") and not line.startswith("end")]

    start, end = None, None
    for line in lines:
        if line.startswith("start") or line.startswith("end"):
            label, x, y = line.strip().split(maxsplit=2)
            x = ''.join(filter(str.isdigit, x))
            y = ''.join(filter(str.isdigit, y))
            if label == "start":
                start = (int(y), int(x))
            elif label == "end":
                end = (int(y), int(x))

    return maze, start, end


def print_maze(maze, start, end, path=[]):
    sx, sy = start
    ex, ey = end
    maze[sx][sy] = "S"
    maze[ex][ey] = "E"
    maze_copy = [row[:] for row in maze]
    for x, y in path:
        maze_copy[x][y] = "o"

    for row in maze_copy:
        print("".join(row))
    print()


def visualize_maze(frames, algorithm_name):
    def preprocess_for_plot(maze):
        return [[
            0 if cell == "X" else
            1 if cell == " " else
            2 if cell == "#" else
            5 if cell == "o" else
            3 if cell == "S" else
            4 if cell == "E" else
            -1 for cell in row
        ] for row in maze]

    # Ensure 'S' and 'E' are present in the first frame
    if "S" not in [cell for row in frames[0] for cell in row]:
        print(maze[start[0]][start[1]], frames[0][start[0]][start[1]])
        raise ValueError("The start position 'S' is not present in the initial frame.")
    if "E" not in [cell for row in frames[0] for cell in row]:
        raise ValueError("The end position 'E' is not present in the initial frame.")

    processed_frames = [preprocess_for_plot(frame) for frame in frames]

    cmap = ListedColormap([
        "black",  # 0: Walls ('X')
        "white",  # 1: Open path spaces (' ')
        "blue",  # 2: Moving path (explored, '#')
        "green",  # 3: Start position ('S')
        "red",  # 4: End position ('E')
        "yellow"  # 5: Final path ('o')
    ])

    fig, ax = plt.subplots()
    im = ax.imshow(processed_frames[0], cmap=cmap, interpolation="nearest")
    ax.set_title(algorithm_name, fontsize=16)
    ax.axis("off")
    def update(frame):
        im.set_array(processed_frames[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(processed_frames), interval=200, repeat=False, blit=True
    )
    plt.show()


def dfs_with_animation(maze, start, end):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    stack = [(start, [start])]
    visited = set()
    opened = -1

    maze_copy = [row[:] for row in maze]
    frames = [maze_copy]

    while stack:
        (x, y), path = stack.pop()
        if (x, y) in visited:
            continue

        visited.add((x, y))
        maze_copy[x][y] = '#'
        opened += 1
        frames.append([row[:] for row in maze_copy])

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[x][y] = 'E'
            frames.append([row[:] for row in maze_copy])
            return frames

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze)
                    and 0 <= ny < len(maze[0])
                    and maze_copy[nx][ny] != "X"
                    and (nx, ny) not in visited
            ):
                stack.append(((nx, ny), path + [(nx, ny)]))
    return frames


def bfs_with_animation(maze, start, end):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)

    maze_copy = [row[:] for row in maze]
    frames = [maze_copy]

    while queue:
        (x, y), path = queue.popleft()

        # if (x, y) == end:
        #     for px, py in path:
        #         maze_copy[px][py] = "o" if (px, py) != start else "S"
        #     maze_copy[x][y] = 'E'
        #     frames.append([row[:] for row in maze_copy])
        #     return frames

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze)
                    and 0 <= ny < len(maze[0])
                    and maze_copy[nx][ny] != "X"
                    and (nx, ny) not in visited
            ):
                queue.append(((nx, ny), path + [(nx, ny)]))
                visited.add((nx, ny))
                maze_copy[nx][ny] = "#"
                frames.append([row[:] for row in maze_copy])
                if (nx, ny) == end:
                    for px, py in path:
                        maze_copy[px][py] = "o" if (px, py) != start else "S"
                    maze_copy[nx][ny] = 'E'
                    frames.append([row[:] for row in maze_copy])
                    return frames

    return frames


def greedy_search_with_animation(maze, start, end):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    pq = [(heuristic(start, end), start, [start])]
    visited = set()
    opened = 0

    maze_copy = [row[:] for row in maze]
    frames = [maze_copy]

    while pq:
        _, (x, y), path = heapq.heappop(pq)
        if (x, y) in visited:
            continue

        visited.add((x, y))
        maze_copy[x][y] = "#"
        opened += 1
        frames.append([row[:] for row in maze_copy])

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[x][y] = 'E'
            frames.append([row[:] for row in maze_copy])
            return frames

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze)
                    and 0 <= ny < len(maze[0])
                    and maze_copy[nx][ny] != "X"
                    and (nx, ny) not in visited
            ):
                heapq.heappush(pq, (heuristic((nx, ny), end), (nx, ny), path + [(nx, ny)]))
    return frames


def a_star_with_animation(maze, start, end):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    pq = [(0, start, [start])]
    g_score = {start: 0}
    opened = 0

    maze_copy = [row[:] for row in maze]
    frames = [maze_copy]

    while pq:
        _, (x, y), path = heapq.heappop(pq)
        maze_copy[x][y] = "#"
        opened += 1
        frames.append([row[:] for row in maze_copy])

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[x][y] = 'E'
            frames.append([row[:] for row in maze_copy])
            return frames

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze)
                    and 0 <= ny < len(maze[0])
                    and maze_copy[nx][ny] != "X"
            ):
                new_g = g_score[(x, y)] + 1
                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f_score = new_g + heuristic((nx, ny), end)
                    heapq.heappush(pq, (f_score, (nx, ny), path + [(nx, ny)]))
    return frames


def random_search_with_animation(maze, start, end):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    path = [start]
    visited = set()
    opened = 0
    visited.add(start)

    maze_copy = [row[:] for row in maze]
    frames = []

    while path:
        x, y = path[-1]

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[start[0]][start[1]] = "S"
            maze_copy[end[0]][end[1]] = "E"
            frames.append([row[:] for row in maze_copy])
            break

        visited.add((x, y))
        maze_copy[x][y] = "#"
        maze_copy[start[0]][start[1]] = "S"
        opened += 1
        frames.append([row[:] for row in maze_copy])

        random.shuffle(directions)
        moved = False

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < len(maze)
                and 0 <= ny < len(maze[0])
                and maze_copy[nx][ny] != "X"
                and (nx, ny) not in visited
            ):
                path.append((nx, ny))
                moved = True
                break

        if not moved:
            path.pop()

        if not frames:
            frames.append([row[:] for row in maze_copy])

    return frames


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])




if __name__ == "__main__":
    maze, start, end = load_maze("maze.txt")

    algorithms = [
        ("DFS", dfs_with_animation),
        ("BFS", bfs_with_animation),
        ("Greedy", greedy_search_with_animation),
        ("A*", a_star_with_animation),
        ("Random Search", random_search_with_animation),
    ]

    for algo_name, algo_func in algorithms:
        frames = algo_func(maze, start, end)
        visualize_maze(frames, algo_name)